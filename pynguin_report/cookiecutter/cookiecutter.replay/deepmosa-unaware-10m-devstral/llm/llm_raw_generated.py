####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_get_file_name():
    # Test with Path object and template name without .json suffix
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


# LLM-generated content at query #4
#--------------------------

```python
def test_dump():
    import tempfile
    import os
    from pathlib import Path

    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        replay_dir = Path(temp_dir)
        template_name = "test_template"
        context = {
            "cookiecutter": {
                "project_name": "test_project",
                "project_slug": "test_project_slug"
            }
        }

        # Call the dump function
        dump(replay_dir, template_name, context)

        # Check if the file was created
        file_path = get_file_name(replay_dir, template_name)
        assert os.path.exists(file_path)

        # Check if the file contains the correct data
        with open(file_path, encoding="utf-8") as infile:
            loaded_context = json.load(infile)

        assert loaded_context == context


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
def test_load():
    # Setup
    replay_dir = "test_replay"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    replay_file = get_file_name(replay_dir, template_name)
    os.makedirs(replay_dir, exist_ok=True)
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test
    loaded_context = load(replay_dir, template_name)

    # Assert
    assert loaded_context == context

    # Cleanup
    os.remove(replay_file)
    os.rmdir(replay_dir)


# LLM-generated content at query #7
#--------------------------

```python
def test_dump():
    # Test successful dump with valid context
    replay_dir = 'test_replay_dir'
    template_name = 'test_template'
    context = {'cookiecutter': {'key': 'value'}}
    dump(replay_dir, template_name, context)
    assert os.path.exists(get_file_name(replay_dir, template_name))

    # Test dump with invalid context (missing cookiecutter key)
    invalid_context = {'key': 'value'}
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        dump(replay_dir, template_name, invalid_context)

    # Test dump with template_name ending in .json
    template_name_json = 'test_template.json'
    dump(replay_dir, template_name_json, context)
    assert os.path.exists(get_file_name(replay_dir, template_name_json))


# LLM-generated content at query #8
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    replay_file = tmp_path / f"{template_name}.json"

    # Create a test file
    with open(replay_file, "w", encoding="utf-8") as f:
        json.dump(context, f, indent=2)

    # Test
    result = load(tmp_path, template_name)

    # Assert
    assert result == context

def test_load_without_json_extension(tmp_path):
    # Setup
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
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
    context = {"key": "value"}
    replay_file = tmp_path / f"{template_name}.json"

    # Create a test file
    with open(replay_file, "w", encoding="utf-8") as f:
        json.dump(context, f, indent=2)

    # Test and Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(tmp_path, template_name)

def test_load_file_not_found(tmp_path):
    # Setup
    template_name = "nonexistent_template"

    # Test and Assert
    with pytest.raises(FileNotFoundError):
        load(tmp_path, template_name)


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
def test_load(tmp_path):
    # Setup
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "Test Project",
            "author": "Test Author"
        }
    }
    replay_file = tmp_path / f"{template_name}.json"
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test
    result = load(tmp_path, template_name)

    # Assert
    assert result == context

def test_load_without_json_extension(tmp_path):
    # Setup
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "Test Project",
            "author": "Test Author"
        }
    }
    replay_file = tmp_path / f"{template_name}.json"
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test
    result = load(tmp_path, template_name)

    # Assert
    assert result == context

def test_load_missing_cookiecutter_key(tmp_path):
    # Setup
    template_name = "test_template"
    context = {
        "project_name": "Test Project",
        "author": "Test Author"
    }
    replay_file = tmp_path / f"{template_name}.json"
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test and Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(tmp_path, template_name)


# LLM-generated content at query #11
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
    with open(expected_file, encoding="utf-8") as infile:
        loaded_context = json.load(infile)
    assert loaded_context == context

    # Cleanup
    os.remove(expected_file)
    os.rmdir(replay_dir)


# LLM-generated content at query #12
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    replay_dir = tmp_path
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

def test_load_missing_cookiecutter_key(tmp_path):
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    context = {"project_name": "test_project"}  # Missing 'cookiecutter' key
    replay_file = get_file_name(replay_dir, template_name)

    # Create a test file
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test and Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


# LLM-generated content at query #13
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


# LLM-generated content at query #14
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


# LLM-generated content at query #15
#--------------------------

```python
def test_dump():
    # Test dump function
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}

    # Call the function
    dump(replay_dir, template_name, context)

    # Check if file was created
    file_name = get_file_name(replay_dir, template_name)
    assert os.path.exists(file_name)

    # Check if file content is correct
    with open(file_name, encoding="utf-8") as infile:
        loaded_context = json.load(infile)
    assert loaded_context == context

    # Clean up
    os.remove(file_name)
    os.rmdir(replay_dir)


# LLM-generated content at query #16
#--------------------------

```python
def test_dump():
    # Setup
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}

    # Test
    dump(replay_dir, template_name, context)

    # Assert
    replay_file = get_file_name(replay_dir, template_name)
    assert os.path.exists(replay_file)

    with open(replay_file, encoding="utf-8") as infile:
        loaded_context = json.load(infile)

    assert loaded_context == context

    # Cleanup
    os.remove(replay_file)
    os.rmdir(replay_dir)


# LLM-generated content at query #17
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


# LLM-generated content at query #18
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


# LLM-generated content at query #19
#--------------------------

```python
def test_dump():
    # Setup
    import tempfile
    import os
    from pathlib import Path

    replay_dir = tempfile.mkdtemp()
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "repo_name": "test_repo"
        }
    }

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


# LLM-generated content at query #20
#--------------------------

```python
def test_load():
    # Setup
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    replay_file = get_file_name(replay_dir, template_name)

    # Create the replay directory and file
    make_sure_path_exists(replay_dir)
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test
    loaded_context = load(replay_dir, template_name)

    # Assert
    assert loaded_context == context

    # Cleanup
    os.remove(replay_file)
    os.rmdir(replay_dir)


# LLM-generated content at query #21
#--------------------------

```python
def test_dump():
    # Setup
    replay_dir = "test_replay"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    expected_file = os.path.join(replay_dir, f"{template_name}.json")

    # Exercise
    dump(replay_dir, template_name, context)

    # Verify
    assert os.path.exists(expected_file)
    with open(expected_file, encoding="utf-8") as infile:
        loaded_context = json.load(infile)
    assert loaded_context == context

    # Cleanup
    os.remove(expected_file)
    os.rmdir(replay_dir)


# LLM-generated content at query #22
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


# LLM-generated content at query #23
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


# LLM-generated content at query #24
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    replay_file = tmp_path / f"{template_name}.json"

    # Create a test file
    with open(replay_file, "w", encoding="utf-8") as f:
        json.dump(context, f, indent=2)

    # Test
    result = load(tmp_path, template_name)

    # Assert
    assert result == context

def test_load_without_json_extension(tmp_path):
    # Setup
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
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
    context = {"key": "value"}
    replay_file = tmp_path / f"{template_name}.json"

    # Create a test file
    with open(replay_file, "w", encoding="utf-8") as f:
        json.dump(context, f, indent=2)

    # Test and Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(tmp_path, template_name)


# LLM-generated content at query #25
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

    # Test error when cookiecutter key is missing
    invalid_context = {"key": "value"}
    invalid_replay_file = get_file_name(replay_dir, "invalid_template")
    with open(invalid_replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(invalid_context, outfile, indent=2)

    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, "invalid_template")


# LLM-generated content at query #26
#--------------------------

```python
def test_dump():
    # Setup
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    expected_file = os.path.join(replay_dir, f"{template_name}.json")

    # Exercise
    dump(replay_dir, template_name, context)

    # Verify
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
def test_load(tmp_path):
    # Setup test data
    test_context = {
        'cookiecutter': {
            'project_name': 'test_project',
            'author': 'test_author'
        }
    }
    template_name = 'test_template'
    replay_dir = tmp_path / 'replay'
    replay_dir.mkdir()

    # Create a test file
    replay_file = replay_dir / f'{template_name}.json'
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(test_context, f, indent=2)

    # Test the load function
    loaded_context = load(replay_dir, template_name)

    # Assertions
    assert loaded_context == test_context
    assert loaded_context['cookiecutter']['project_name'] == 'test_project'
    assert loaded_context['cookiecutter']['author'] == 'test_author'

    # Test with missing cookiecutter key
    invalid_context = {'invalid_key': 'value'}
    invalid_file = replay_dir / 'invalid.json'
    with open(invalid_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f, indent=2)

    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, 'invalid')


# LLM-generated content at query #28
#--------------------------

```python
def test_dump(tmp_path):
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}

    dump(replay_dir, template_name, context)

    replay_file = replay_dir / f"{template_name}.json"
    assert replay_file.exists()

    with open(replay_file, "r", encoding="utf-8") as infile:
        loaded_context = json.load(infile)

    assert loaded_context == context

def test_dump_with_json_suffix():
    replay_dir = "test_replay"
    template_name = "test_template.json"
    context = {"cookiecutter": {"key": "value"}}

    dump(replay_dir, template_name, context)

    replay_file = os.path.join(replay_dir, template_name)
    assert os.path.exists(replay_file)

    with open(replay_file, "r", encoding="utf-8") as infile:
        loaded_context = json.load(infile)

    assert loaded_context == context

def test_dump_missing_cookiecutter_key():
    replay_dir = "test_replay"
    template_name = "test_template"
    context = {"key": "value"}

    with pytest.raises(ValueError) as excinfo:
        dump(replay_dir, template_name, context)

    assert "Context is required to contain a cookiecutter key" in str(excinfo.value)


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
    result = load(replay_dir, template_name)

    # Assert
    assert result == context

def test_load_missing_cookiecutter_key(tmp_path):
    # Setup
    template_name = "test_template"
    context = {"key": "value"}  # Missing 'cookiecutter' key
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()

    # Write test file
    replay_file = get_file_name(replay_dir, template_name)
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test & Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)

def test_load_with_json_extension(tmp_path):
    # Setup
    template_name = "test_template.json"
    context = {"cookiecutter": {"key": "value"}}
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()

    # Write test file
    replay_file = get_file_name(replay_dir, template_name)
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test
    result = load(replay_dir, template_name)

    # Assert
    assert result == context


# LLM-generated content at query #31
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


# LLM-generated content at query #32
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test_project"}}
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    replay_file = get_file_name(replay_dir, template_name)

    # Write test data
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test
    loaded_context = load(replay_dir, template_name)

    # Assert
    assert loaded_context == context


# LLM-generated content at query #33
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    test_context = {'cookiecutter': {'project_name': 'test_project'}}
    template_name = 'test_template'
    replay_dir = tmp_path
    replay_file = get_file_name(replay_dir, template_name)

    # Create a test file
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(test_context, outfile, indent=2)

    # Test
    result = load(replay_dir, template_name)

    # Assert
    assert result == test_context


# LLM-generated content at query #34
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    test_dir = tmp_path / "replay"
    test_dir.mkdir()
    test_file = test_dir / "test.json"
    context = {"cookiecutter": {"project_name": "test"}}
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(context, f, indent=2)

    # Test
    result = load(test_dir, "test")

    # Assert
    assert result == context

def test_load_missing_cookiecutter_key(tmp_path):
    # Setup
    test_dir = tmp_path / "replay"
    test_dir.mkdir()
    test_file = test_dir / "test.json"
    context = {"project_name": "test"}
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(context, f, indent=2)

    # Test & Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(test_dir, "test")

def test_load_file_not_found(tmp_path):
    # Setup
    test_dir = tmp_path / "replay"
    test_dir.mkdir()

    # Test & Assert
    with pytest.raises(FileNotFoundError):
        load(test_dir, "nonexistent")


# LLM-generated content at query #35
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


# LLM-generated content at query #36
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test_project"}}

    # Create a test file
    replay_file = get_file_name(replay_dir, template_name)
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test
    loaded_context = load(replay_dir, template_name)

    # Assert
    assert loaded_context == context

    # Test with .json suffix
    template_name_json = "test_template.json"
    replay_file_json = get_file_name(replay_dir, template_name_json)
    with open(replay_file_json, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    loaded_context_json = load(replay_dir, template_name_json)
    assert loaded_context_json == context

    # Test error handling
    invalid_template_name = "invalid_template"
    invalid_replay_file = get_file_name(replay_dir, invalid_template_name)
    with open(invalid_replay_file, 'w', encoding="utf-8") as outfile:
        json.dump({"invalid_key": "value"}, outfile, indent=2)

    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, invalid_template_name)


# LLM-generated content at query #37
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test_project"}}
    replay_file = tmp_path / f"{template_name}.json"

    # Create a test file
    with open(replay_file, "w", encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test
    result = load(tmp_path, template_name)

    # Assert
    assert result == context

def test_load_without_json_extension(tmp_path):
    # Setup
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test_project"}}
    replay_file = tmp_path / f"{template_name}.json"

    # Create a test file
    with open(replay_file, "w", encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test
    result = load(tmp_path, template_name)

    # Assert
    assert result == context

def test_load_missing_cookiecutter_key(tmp_path):
    # Setup
    template_name = "test_template"
    context = {"project_name": "test_project"}
    replay_file = tmp_path / f"{template_name}.json"

    # Create a test file
    with open(replay_file, "w", encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test and Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(tmp_path, template_name)


# LLM-generated content at query #38
#--------------------------

```python
def test_load():
    # Setup
    replay_dir = "test_replay"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    replay_file = get_file_name(replay_dir, template_name)
    os.makedirs(replay_dir, exist_ok=True)
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test
    result = load(replay_dir, template_name)

    # Assert
    assert result == context

    # Cleanup
    os.remove(replay_file)
    os.rmdir(replay_dir)


# LLM-generated content at query #39
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

    # Execute
    dump(replay_dir, template_name, context)

    # Verify
    expected_file = os.path.join(replay_dir, f"{template_name}.json")
    assert os.path.exists(expected_file)

    with open(expected_file, "r", encoding="utf-8") as f:
        loaded_context = json.load(f)

    assert loaded_context == context

    # Cleanup
    os.remove(expected_file)
    os.rmdir(replay_dir)


# LLM-generated content at query #40
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


# LLM-generated content at query #41
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


# LLM-generated content at query #42
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test_project"}}
    dump(replay_dir, template_name, context)

    # Test
    loaded_context = load(replay_dir, template_name)

    # Assert
    assert loaded_context == context


# LLM-generated content at query #43
#--------------------------

```python
def test_dump():
    # Setup
    import tempfile
    from pathlib import Path
    replay_dir = tempfile.mkdtemp()
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test_project"}}

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


# LLM-generated content at query #44
#--------------------------

```python
def test_dump(tmp_path):
    # Test basic dump functionality
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test_project"}}

    dump(replay_dir, template_name, context)

    # Verify file was created with correct content
    replay_file = get_file_name(replay_dir, template_name)
    assert os.path.exists(replay_file)

    with open(replay_file, encoding="utf-8") as f:
        loaded_context = json.load(f)

    assert loaded_context == context

    # Test with .json suffix in template name
    template_name_json = "test_template.json"
    dump(replay_dir, template_name_json, context)

    replay_file_json = get_file_name(replay_dir, template_name_json)
    assert os.path.exists(replay_file_json)

    with open(replay_file_json, encoding="utf-8") as f:
        loaded_context_json = json.load(f)

    assert loaded_context_json == context

    # Test error when context doesn't contain cookiecutter key
    invalid_context = {"project_name": "test_project"}
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, invalid_context)


# LLM-generated content at query #45
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


# LLM-generated content at query #46
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


# LLM-generated content at query #47
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


# LLM-generated content at query #48
#--------------------------

```python
def test_dump(tmp_path):
    # Test basic functionality
    test_context = {'cookiecutter': {'project_name': 'test'}}
    dump(tmp_path, 'test', test_context)

    # Verify file was created
    expected_file = tmp_path / 'test.json'
    assert expected_file.exists()

    # Verify content
    with open(expected_file, 'r', encoding="utf-8") as f:
        content = json.load(f)
    assert content == test_context

    # Test with .json extension
    dump(tmp_path, 'test.json', test_context)
    expected_file = tmp_path / 'test.json'
    assert expected_file.exists()

    # Test ValueError when cookiecutter key is missing
    with pytest.raises(ValueError):
        dump(tmp_path, 'test', {'invalid': 'context'})

    # Test directory creation
    non_existent_dir = tmp_path / 'non_existent'
    dump(non_existent_dir, 'test', test_context)
    assert (non_existent_dir / 'test.json').exists()


# LLM-generated content at query #49
#--------------------------

```python
def test_load():
    # Setup
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    replay_file = get_file_name(replay_dir, template_name)
    os.makedirs(replay_dir, exist_ok=True)
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test
    loaded_context = load(replay_dir, template_name)

    # Assert
    assert loaded_context == context

    # Teardown
    os.remove(replay_file)
    os.rmdir(replay_dir)


# LLM-generated content at query #50
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


# LLM-generated content at query #51
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


# LLM-generated content at query #52
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


# LLM-generated content at query #53
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    test_context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "test_author"
        }
    }
    template_name = "test_template"
    replay_dir = tmp_path
    replay_file = get_file_name(replay_dir, template_name)

    # Create a test file
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(test_context, outfile, indent=2)

    # Test
    result = load(replay_dir, template_name)

    # Assert
    assert result == test_context

def test_load_missing_cookiecutter_key(tmp_path):
    # Setup
    test_context = {
        "project_name": "test_project",
        "author": "test_author"
    }
    template_name = "test_template"
    replay_dir = tmp_path
    replay_file = get_file_name(replay_dir, template_name)

    # Create a test file without cookiecutter key
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(test_context, outfile, indent=2)

    # Test and Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)

def test_load_with_json_suffix(tmp_path):
    # Setup
    test_context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "test_author"
        }
    }
    template_name = "test_template.json"
    replay_dir = tmp_path
    replay_file = get_file_name(replay_dir, template_name)

    # Create a test file
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(test_context, outfile, indent=2)

    # Test
    result = load(replay_dir, template_name)

    # Assert
    assert result == test_context


# LLM-generated content at query #54
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    replay_file = tmp_path / f"{template_name}.json"

    # Create a test file
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test
    result = load(tmp_path, template_name)

    # Assert
    assert result == context

def test_load_without_json_extension(tmp_path):
    # Setup
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    replay_file = tmp_path / f"{template_name}.json"

    # Create a test file
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test
    result = load(tmp_path, template_name)

    # Assert
    assert result == context

def test_load_missing_cookiecutter_key(tmp_path):
    # Setup
    template_name = "test_template"
    context = {"key": "value"}
    replay_file = tmp_path / f"{template_name}.json"

    # Create a test file
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test and Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(tmp_path, template_name)


# LLM-generated content at query #55
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


# LLM-generated content at query #56
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    template_name = "test_template"
    replay_dir = tmp_path
    context = {"cookiecutter": {"key": "value"}}

    # Create a test file
    replay_file = get_file_name(replay_dir, template_name)
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test
    loaded_context = load(replay_dir, template_name)

    # Assert
    assert loaded_context == context


# LLM-generated content at query #57
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


# LLM-generated content at query #58
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


# LLM-generated content at query #59
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


# LLM-generated content at query #60
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


# LLM-generated content at query #61
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


# LLM-generated content at query #62
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


# LLM-generated content at query #63
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    replay_file = tmp_path / f"{template_name}.json"
    replay_file.write_text(json.dumps(context))

    # Test
    result = load(tmp_path, template_name)

    # Assert
    assert result == context


# LLM-generated content at query #64
#--------------------------

```python
def test_dump(tmp_path):
    # Test basic dump functionality
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test_project"}}

    dump(replay_dir, template_name, context)

    # Verify the file was created with correct content
    replay_file = get_file_name(replay_dir, template_name)
    assert os.path.exists(replay_file)

    with open(replay_file, encoding="utf-8") as f:
        loaded_context = json.load(f)

    assert loaded_context == context

    # Test with .json suffix in template_name
    template_name_json = "test_template.json"
    dump(replay_dir, template_name_json, context)

    replay_file_json = get_file_name(replay_dir, template_name_json)
    assert os.path.exists(replay_file_json)

    # Test ValueError when cookiecutter key is missing
    invalid_context = {"project_name": "test_project"}
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, invalid_context)


# LLM-generated content at query #65
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

    expected_file = replay_dir / f"{template_name}.json"
    assert expected_file.exists()

    with open(expected_file, encoding="utf-8") as infile:
        loaded_context = json.load(infile)

    assert loaded_context == context

def test_dump_with_json_suffix():
    replay_dir = "test_replay"
    template_name = "test_template.json"
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "test_author"
        }
    }

    dump(replay_dir, template_name, context)

    expected_file = os.path.join(replay_dir, template_name)
    assert os.path.exists(expected_file)

    with open(expected_file, encoding="utf-8") as infile:
        loaded_context = json.load(infile)

    assert loaded_context == context

def test_dump_missing_cookiecutter_key():
    replay_dir = "test_replay"
    template_name = "test_template"
    context = {
        "project_name": "test_project",
        "author": "test_author"
    }

    with pytest.raises(ValueError) as excinfo:
        dump(replay_dir, template_name, context)

    assert "Context is required to contain a cookiecutter key" in str(excinfo.value)


# LLM-generated content at query #66
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


# LLM-generated content at query #67
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
        # Exercise
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


# LLM-generated content at query #68
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
    assert replay_file.exists()

    with open(replay_file, encoding="utf-8") as infile:
        loaded_context = json.load(infile)

    assert loaded_context == context


# LLM-generated content at query #69
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    template_name = "test_template"
    replay_dir = tmp_path
    context = {"cookiecutter": {"key": "value"}}

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
    replay_dir = tmp_path
    context = {"key": "value"}  # Missing 'cookiecutter' key

    # Create a test file
    replay_file = get_file_name(replay_dir, template_name)
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test and Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


# LLM-generated content at query #70
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    template_name = "test_template"
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    context = {"cookiecutter": {"project_name": "test_project"}}
    replay_file = replay_dir / f"{template_name}.json"

    # Write test data
    with open(replay_file, "w", encoding="utf-8") as f:
        json.dump(context, f, indent=2)

    # Test
    result = load(replay_dir, template_name)

    # Assert
    assert result == context


# LLM-generated content at query #71
#--------------------------

```python
def test_dump(tmp_path):
    # Test successful dump
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}

    dump(replay_dir, template_name, context)

    expected_file = replay_dir / f"{template_name}.json"
    assert expected_file.exists()

    with open(expected_file, encoding="utf-8") as f:
        loaded_context = json.load(f)

    assert loaded_context == context

    # Test with .json suffix in template_name
    template_name_with_suffix = "test_template.json"
    dump(replay_dir, template_name_with_suffix, context)

    expected_file_with_suffix = replay_dir / template_name_with_suffix
    assert expected_file_with_suffix.exists()

    # Test missing cookiecutter key
    invalid_context = {"key": "value"}
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, invalid_context)


# LLM-generated content at query #72
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


# LLM-generated content at query #73
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


# LLM-generated content at query #74
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


# LLM-generated content at query #75
#--------------------------

```python
def test_dump():
    # Setup
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    expected_file = os.path.join(replay_dir, "test_template.json")

    # Exercise
    dump(replay_dir, template_name, context)

    # Verify
    assert os.path.exists(expected_file)
    with open(expected_file, encoding="utf-8") as f:
        loaded_context = json.load(f)
    assert loaded_context == context

    # Cleanup
    os.remove(expected_file)
    os.rmdir(replay_dir)


# LLM-generated content at query #76
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test_project"}}
    replay_file = tmp_path / f"{template_name}.json"

    # Create a test file
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test
    result = load(tmp_path, template_name)

    # Assert
    assert result == context


# LLM-generated content at query #77
#--------------------------

```python
def test_dump():
    # Setup
    replay_dir = "test_replay"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    expected_file = os.path.join(replay_dir, f"{template_name}.json")

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


# LLM-generated content at query #78
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    replay_file = get_file_name(replay_dir, template_name)

    # Create the replay directory and file
    replay_dir.mkdir()
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test
    loaded_context = load(replay_dir, template_name)

    # Assert
    assert loaded_context == context

def test_load_missing_cookiecutter_key(tmp_path):
    # Setup
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"key": "value"}  # Missing 'cookiecutter' key
    replay_file = get_file_name(replay_dir, template_name)

    # Create the replay directory and file
    replay_dir.mkdir()
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test and Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


# LLM-generated content at query #79
#--------------------------

```python
def test_load():
    # Setup
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    replay_file = get_file_name(replay_dir, template_name)
    os.makedirs(replay_dir, exist_ok=True)
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test
    result = load(replay_dir, template_name)

    # Assert
    assert result == context

    # Cleanup
    os.remove(replay_file)
    os.rmdir(replay_dir)


# LLM-generated content at query #80
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


# LLM-generated content at query #81
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    test_context = {'cookiecutter': {'project_name': 'test_project'}}
    template_name = 'test_template'
    replay_dir = tmp_path
    replay_file = get_file_name(replay_dir, template_name)

    # Create a test file
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(test_context, outfile, indent=2)

    # Test
    result = load(replay_dir, template_name)

    # Assert
    assert result == test_context


# LLM-generated content at query #82
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
            "author": "test_author"
        }
    }
    replay_file = get_file_name(replay_dir, template_name)
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test
    loaded_context = load(replay_dir, template_name)

    # Assert
    assert loaded_context == context


# LLM-generated content at query #83
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test_project"}}

    # Write test file
    replay_file = get_file_name(replay_dir, template_name)
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test
    loaded_context = load(replay_dir, template_name)

    # Assert
    assert loaded_context == context


# LLM-generated content at query #84
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


# LLM-generated content at query #85
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


# LLM-generated content at query #86
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


# LLM-generated content at query #87
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


# LLM-generated content at query #88
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    test_context = {'cookiecutter': {'project_name': 'test_project'}}
    test_template_name = 'test_template'
    replay_dir = tmp_path / 'replay'
    replay_dir.mkdir()

    # Write test data
    replay_file = get_file_name(replay_dir, test_template_name)
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(test_context, outfile, indent=2)

    # Test
    result = load(replay_dir, test_template_name)

    # Assert
    assert result == test_context

def test_load_missing_cookiecutter_key(tmp_path):
    # Setup
    test_context = {'project_name': 'test_project'}  # Missing 'cookiecutter' key
    test_template_name = 'test_template'
    replay_dir = tmp_path / 'replay'
    replay_dir.mkdir()

    # Write test data
    replay_file = get_file_name(replay_dir, test_template_name)
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(test_context, outfile, indent=2)

    # Test and Assert
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, test_template_name)


# LLM-generated content at query #89
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


# LLM-generated content at query #90
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test_project"}}
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()

    # Create a test file
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, "w", encoding="utf-8") as f:
        json.dump(context, f, indent=2)

    # Test
    loaded_context = load(replay_dir, template_name)

    # Assert
    assert loaded_context == context


# LLM-generated content at query #91
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


# LLM-generated content at query #92
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


# LLM-generated content at query #93
#--------------------------

```python
def test_dump():
    # Setup
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    expected_file_path = os.path.join(replay_dir, f"{template_name}.json")

    # Exercise
    dump(replay_dir, template_name, context)

    # Verify
    assert os.path.exists(expected_file_path)
    with open(expected_file_path, encoding="utf-8") as infile:
        loaded_context = json.load(infile)
    assert loaded_context == context

    # Cleanup
    os.remove(expected_file_path)
    os.rmdir(replay_dir)


# LLM-generated content at query #94
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


# LLM-generated content at query #95
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

    # Teardown
    os.remove(get_file_name(replay_dir, template_name))
    os.rmdir(replay_dir)


# LLM-generated content at query #96
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


# LLM-generated content at query #97
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


# LLM-generated content at query #98
#--------------------------

```python
def test_dump(tmp_path):
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}

    dump(replay_dir, template_name, context)

    expected_file = replay_dir / f"{template_name}.json"
    assert expected_file.exists()

    with open(expected_file, encoding="utf-8") as f:
        loaded_context = json.load(f)

    assert loaded_context == context

def test_dump_with_json_extension():
    replay_dir = "test_replay"
    template_name = "test_template.json"
    context = {"cookiecutter": {"key": "value"}}

    dump(replay_dir, template_name, context)

    expected_file = os.path.join(replay_dir, template_name)
    assert os.path.exists(expected_file)

    with open(expected_file, encoding="utf-8") as f:
        loaded_context = json.load(f)

    assert loaded_context == context

def test_dump_missing_cookiecutter_key():
    replay_dir = "test_replay"
    template_name = "test_template"
    context = {"key": "value"}

    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


# LLM-generated content at query #99
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    template_name = "test_template"
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    context = {"cookiecutter": {"project_name": "test_project"}}
    replay_file = replay_dir / f"{template_name}.json"

    # Write test data
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test
    loaded_context = load(replay_dir, template_name)

    # Assert
    assert loaded_context == context


# LLM-generated content at query #100
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    test_context = {
        'cookiecutter': {
            'project_name': 'test_project',
            'author': 'test_author'
        },
        'other_data': 'some_value'
    }
    template_name = 'test_template'
    replay_dir = tmp_path
    replay_file = get_file_name(replay_dir, template_name)

    # Create a test file
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(test_context, outfile, indent=2)

    # Test
    loaded_context = load(replay_dir, template_name)

    # Assert
    assert loaded_context == test_context

    # Test with missing cookiecutter key
    invalid_context = {'other_data': 'some_value'}
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(invalid_context, outfile, indent=2)

    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, template_name)


# LLM-generated content at query #101
#--------------------------

```python
def test_dump():
    # Test basic functionality
    replay_dir = "test_replay"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}

    dump(replay_dir, template_name, context)

    # Verify file was created
    expected_file = get_file_name(replay_dir, template_name)
    assert os.path.exists(expected_file)

    # Verify content
    with open(expected_file, encoding="utf-8") as f:
        loaded_context = json.load(f)
    assert loaded_context == context

    # Test with .json extension
    template_name_json = "test_template.json"
    dump(replay_dir, template_name_json, context)

    expected_file_json = get_file_name(replay_dir, template_name_json)
    assert os.path.exists(expected_file_json)

    # Test error when cookiecutter key is missing
    invalid_context = {"key": "value"}
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, invalid_context)

    # Cleanup
    if os.path.exists(replay_dir):
        os.remove(expected_file)
        if os.path.exists(expected_file_json):
            os.remove(expected_file_json)
        os.rmdir(replay_dir)


# LLM-generated content at query #102
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


# LLM-generated content at query #103
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    test_context = {'cookiecutter': {'project_name': 'test_project'}}
    template_name = 'test_template'
    replay_dir = tmp_path
    replay_file = get_file_name(replay_dir, template_name)

    # Create a test file
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(test_context, outfile, indent=2)

    # Test
    result = load(replay_dir, template_name)

    # Assert
    assert result == test_context

    # Test with .json suffix
    template_name_json = 'test_template.json'
    replay_file_json = get_file_name(replay_dir, template_name_json)

    with open(replay_file_json, 'w', encoding="utf-8") as outfile:
        json.dump(test_context, outfile, indent=2)

    result_json = load(replay_dir, template_name_json)
    assert result_json == test_context

    # Test missing cookiecutter key
    invalid_context = {'project_name': 'test_project'}
    replay_file_invalid = get_file_name(replay_dir, 'invalid_template')

    with open(replay_file_invalid, 'w', encoding="utf-8") as outfile:
        json.dump(invalid_context, outfile, indent=2)

    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, 'invalid_template')


# LLM-generated content at query #104
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
    with open(expected_file, encoding="utf-8") as infile:
        loaded_context = json.load(infile)
    assert loaded_context == context

    # Cleanup
    os.remove(expected_file)
    os.rmdir(replay_dir)


# LLM-generated content at query #105
#--------------------------

```python
def test_load():
    # Setup
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    replay_file = get_file_name(replay_dir, template_name)

    # Create the directory and file for testing
    make_sure_path_exists(replay_dir)
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test
    loaded_context = load(replay_dir, template_name)

    # Assert
    assert loaded_context == context

    # Cleanup
    os.remove(replay_file)
    os.rmdir(replay_dir)


# LLM-generated content at query #106
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


# LLM-generated content at query #107
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


# LLM-generated content at query #108
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


# LLM-generated content at query #109
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

    # Execute
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


# LLM-generated content at query #110
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


# LLM-generated content at query #111
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
        # Exercise
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


# LLM-generated content at query #112
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test_project"}}
    replay_file = tmp_path / f"{template_name}.json"

    # Create a test file
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test
    result = load(tmp_path, template_name)

    # Assert
    assert result == context


# LLM-generated content at query #113
#--------------------------

```python
def test_dump(tmp_path):
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test_project"}}

    dump(replay_dir, template_name, context)

    expected_file = replay_dir / f"{template_name}.json"
    assert expected_file.exists()

    with open(expected_file, encoding="utf-8") as f:
        loaded_context = json.load(f)

    assert loaded_context == context


# LLM-generated content at query #114
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
    replay_dir = tmp_path
    replay_file = get_file_name(replay_dir, template_name)

    # Write test data to file
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(test_context, outfile, indent=2)

    # Test
    result = load(replay_dir, template_name)

    # Assert
    assert result == test_context

    # Test with missing cookiecutter key
    invalid_context = {'other_key': 'value'}
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(invalid_context, outfile, indent=2)

    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, template_name)


# LLM-generated content at query #115
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


# LLM-generated content at query #116
#--------------------------

```python
def test_dump(tmp_path):
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}

    dump(replay_dir, template_name, context)

    replay_file = get_file_name(replay_dir, template_name)
    assert os.path.exists(replay_file)

    with open(replay_file, encoding="utf-8") as infile:
        loaded_context = json.load(infile)

    assert loaded_context == context

def test_dump_missing_cookiecutter_key(tmp_path):
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"key": "value"}

    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


# LLM-generated content at query #117
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


# LLM-generated content at query #118
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


# LLM-generated content at query #119
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


# LLM-generated content at query #120
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


# LLM-generated content at query #121
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


# LLM-generated content at query #122
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


# LLM-generated content at query #123
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
    with open(replay_file, "w", encoding="utf-8") as f:
        json.dump(context, f, indent=2)

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
    with open(replay_file, "w", encoding="utf-8") as f:
        json.dump(context, f, indent=2)

    # Test and Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


# LLM-generated content at query #124
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

    # Execute
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


# LLM-generated content at query #125
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    replay_file = tmp_path / f"{template_name}.json"
    replay_file.write_text(json.dumps(context))

    # Test
    result = load(tmp_path, template_name)

    # Assert
    assert result == context


# LLM-generated content at query #126
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    template_name = "test_template"
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()

    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "project_slug": "test_project_slug"
        }
    }

    replay_file = get_file_name(replay_dir, template_name)
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test
    loaded_context = load(replay_dir, template_name)

    # Assert
    assert loaded_context == context


# LLM-generated content at query #127
#--------------------------

```python
def test_dump():
    # Setup
    import tempfile
    import os
    from pathlib import Path

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
    expected_file = os.path.join(replay_dir, f"{template_name}.json")
    assert os.path.exists(expected_file)

    with open(expected_file, encoding="utf-8") as infile:
        loaded_context = json.load(infile)

    assert loaded_context == context

    # Cleanup
    os.remove(expected_file)
    os.rmdir(replay_dir)


# LLM-generated content at query #128
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

    # Test with .json suffix
    replay_file_json = get_file_name(replay_dir, f"{template_name}.json")
    with open(replay_file_json, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    loaded_context_json = load(replay_dir, f"{template_name}.json")
    assert loaded_context_json == context

    # Test error case - missing cookiecutter key
    invalid_context = {"project_name": "test_project"}
    invalid_replay_file = get_file_name(replay_dir, "invalid_template")
    with open(invalid_replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(invalid_context, outfile, indent=2)

    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, "invalid_template")


# LLM-generated content at query #129
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
    assert replay_file.exists()

    with open(replay_file, encoding="utf-8") as infile:
        loaded_context = json.load(infile)

    assert loaded_context == context


# LLM-generated content at query #130
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


# LLM-generated content at query #131
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


# LLM-generated content at query #132
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
    assert replay_file.exists()

    with open(replay_file, encoding="utf-8") as infile:
        loaded_context = json.load(infile)

    assert loaded_context == context


# LLM-generated content at query #133
#--------------------------

```python
def test_dump():
    # Setup
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    expected_file = os.path.join(replay_dir, f"{template_name}.json")

    # Exercise
    dump(replay_dir, template_name, context)

    # Verify
    assert os.path.exists(expected_file)
    with open(expected_file, encoding="utf-8") as infile:
        loaded_context = json.load(infile)
    assert loaded_context == context

    # Cleanup
    os.remove(expected_file)
    os.rmdir(replay_dir)


# LLM-generated content at query #134
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


# LLM-generated content at query #135
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


# LLM-generated content at query #136
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    template_name = "test_template"
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()

    test_context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "test_author"
        }
    }

    # Create a test file
    replay_file = get_file_name(replay_dir, template_name)
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(test_context, outfile, indent=2)

    # Test
    result = load(replay_dir, template_name)

    # Assert
    assert result == test_context

def test_load_missing_cookiecutter_key(tmp_path):
    # Setup
    template_name = "test_template"
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()

    test_context = {
        "project_name": "test_project",
        "author": "test_author"
    }

    # Create a test file without cookiecutter key
    replay_file = get_file_name(replay_dir, template_name)
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(test_context, outfile, indent=2)

    # Test and Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)

def test_load_file_not_found(tmp_path):
    # Setup
    template_name = "nonexistent_template"
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()

    # Test and Assert
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


# LLM-generated content at query #137
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


# LLM-generated content at query #138
#--------------------------

```python
def test_dump():
    # Setup
    import tempfile
    from pathlib import Path

    replay_dir = tempfile.mkdtemp()
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test_project"}}

    # Execute
    dump(replay_dir, template_name, context)

    # Verify
    replay_file = get_file_name(replay_dir, template_name)
    assert Path(replay_file).exists()

    with open(replay_file, encoding="utf-8") as infile:
        loaded_context = json.load(infile)

    assert loaded_context == context


# LLM-generated content at query #139
#--------------------------

```python
def test_dump():
    # Setup
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    expected_file_path = os.path.join(replay_dir, f"{template_name}.json")

    # Test
    dump(replay_dir, template_name, context)

    # Assert
    assert os.path.exists(expected_file_path)
    with open(expected_file_path, encoding="utf-8") as infile:
        loaded_context = json.load(infile)
    assert loaded_context == context

    # Cleanup
    os.remove(expected_file_path)
    os.rmdir(replay_dir)


# LLM-generated content at query #140
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


# LLM-generated content at query #141
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


# LLM-generated content at query #142
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

def test_load_file_not_found(tmp_path):
    # Setup
    template_name = "non_existent_template"

    # Test and Assert
    with pytest.raises(FileNotFoundError):
        load(tmp_path, template_name)


# LLM-generated content at query #143
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test_project"}}
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    replay_file = get_file_name(replay_dir, template_name)

    # Write test data
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test
    loaded_context = load(replay_dir, template_name)

    # Assert
    assert loaded_context == context


# LLM-generated content at query #144
#--------------------------

```python
def test_dump(tmp_path):
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "project_slug": "test_project_slug",
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
    }

    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


# LLM-generated content at query #145
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


# LLM-generated content at query #146
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


# LLM-generated content at query #147
#--------------------------

```python
def test_dump(tmp_path):
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}

    dump(replay_dir, template_name, context)

    expected_file = replay_dir / f"{template_name}.json"
    assert expected_file.exists()

    with open(expected_file, "r", encoding="utf-8") as f:
        loaded_context = json.load(f)

    assert loaded_context == context

def test_dump_with_json_suffix():
    replay_dir = "test_replay"
    template_name = "test_template.json"
    context = {"cookiecutter": {"key": "value"}}

    dump(replay_dir, template_name, context)

    expected_file = os.path.join(replay_dir, template_name)
    assert os.path.exists(expected_file)

    with open(expected_file, "r", encoding="utf-8") as f:
        loaded_context = json.load(f)

    assert loaded_context == context

def test_dump_missing_cookiecutter_key():
    replay_dir = "test_replay"
    template_name = "test_template"
    context = {"key": "value"}

    with pytest.raises(ValueError) as excinfo:
        dump(replay_dir, template_name, context)

    assert "Context is required to contain a cookiecutter key" in str(excinfo.value)


# LLM-generated content at query #148
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


# LLM-generated content at query #149
#--------------------------

```python
def test_dump():
    # Setup
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    expected_file = os.path.join(replay_dir, "test_template.json")

    # Exercise
    dump(replay_dir, template_name, context)

    # Verify
    assert os.path.exists(expected_file)
    with open(expected_file, encoding="utf-8") as f:
        loaded_context = json.load(f)
    assert loaded_context == context

    # Cleanup
    os.remove(expected_file)
    os.rmdir(replay_dir)


# LLM-generated content at query #150
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


# LLM-generated content at query #151
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


# LLM-generated content at query #152
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test_project"}}
    replay_file = tmp_path / f"{template_name}.json"

    # Create a test file
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

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
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test and Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(tmp_path, template_name)

def test_load_file_not_found(tmp_path):
    # Setup
    template_name = "nonexistent_template"

    # Test and Assert
    with pytest.raises(FileNotFoundError):
        load(tmp_path, template_name)


# LLM-generated content at query #153
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


# LLM-generated content at query #154
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


# LLM-generated content at query #155
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


# LLM-generated content at query #156
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


# LLM-generated content at query #157
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


# LLM-generated content at query #158
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
        # Exercise
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


# LLM-generated content at query #159
#--------------------------

```python
def test_load():
    # Setup
    replay_dir = "test_replay"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    replay_file = get_file_name(replay_dir, template_name)
    os.makedirs(replay_dir, exist_ok=True)
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test
    loaded_context = load(replay_dir, template_name)

    # Assert
    assert loaded_context == context

    # Cleanup
    os.remove(replay_file)
    os.rmdir(replay_dir)


# LLM-generated content at query #160
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


# LLM-generated content at query #161
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test_project"}}
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()

    # Create a test file
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, "w", encoding="utf-8") as f:
        json.dump(context, f, indent=2)

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
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, "w", encoding="utf-8") as f:
        json.dump(context, f, indent=2)

    # Test and Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)

def test_load_file_not_found(tmp_path):
    # Setup
    template_name = "non_existent_template"
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()

    # Test and Assert
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


# LLM-generated content at query #162
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


# LLM-generated content at query #163
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test_project"}}
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    replay_file = get_file_name(replay_dir, template_name)

    # Create a test file
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test
    loaded_context = load(replay_dir, template_name)

    # Assert
    assert loaded_context == context


# LLM-generated content at query #164
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


# LLM-generated content at query #165
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test_project"}}
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()

    # Write test file
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, "w", encoding="utf-8") as f:
        json.dump(context, f, indent=2)

    # Test
    result = load(replay_dir, template_name)

    # Assert
    assert result == context


# LLM-generated content at query #166
#--------------------------

```python
def test_dump(tmp_path):
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}

    dump(replay_dir, template_name, context)

    replay_file = replay_dir / f"{template_name}.json"
    assert replay_file.exists()

    with open(replay_file, encoding="utf-8") as infile:
        loaded_context = json.load(infile)

    assert loaded_context == context

def test_dump_with_json_suffix():
    replay_dir = "test_replay"
    template_name = "test_template.json"
    context = {"cookiecutter": {"key": "value"}}

    dump(replay_dir, template_name, context)

    replay_file = os.path.join(replay_dir, template_name)
    assert os.path.exists(replay_file)

    with open(replay_file, encoding="utf-8") as infile:
        loaded_context = json.load(infile)

    assert loaded_context == context

def test_dump_missing_cookiecutter_key():
    replay_dir = "test_replay"
    template_name = "test_template"
    context = {"key": "value"}

    with pytest.raises(ValueError) as excinfo:
        dump(replay_dir, template_name, context)

    assert "Context is required to contain a cookiecutter key" in str(excinfo.value)


# LLM-generated content at query #167
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test_project"}}
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    replay_file = get_file_name(replay_dir, template_name)

    # Create a test file
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test
    loaded_context = load(replay_dir, template_name)

    # Assert
    assert loaded_context == context


# LLM-generated content at query #168
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    test_context = {
        'cookiecutter': {
            'project_name': 'test_project',
            'project_slug': 'test_project'
        }
    }
    template_name = 'test_template'
    replay_dir = tmp_path
    replay_file = os.path.join(replay_dir, f'{template_name}.json')

    # Create a test file
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(test_context, outfile, indent=2)

    # Test
    result = load(replay_dir, template_name)

    # Assert
    assert result == test_context


# LLM-generated content at query #169
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    replay_file = tmp_path / f"{template_name}.json"
    replay_file.write_text(json.dumps(context))

    # Test
    result = load(tmp_path, template_name)

    # Assert
    assert result == context


# LLM-generated content at query #170
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    test_context = {
        "cookiecutter": {
            "project_name": "test_project",
            "project_slug": "test_project",
        },
        "other_key": "other_value"
    }
    template_name = "test_template"
    replay_file = tmp_path / f"{template_name}.json"

    # Create a test file
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(test_context, f, indent=2)

    # Test
    result = load(tmp_path, template_name)

    # Assert
    assert result == test_context


# LLM-generated content at query #171
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


# LLM-generated content at query #172
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

def test_load_file_not_found(tmp_path):
    # Setup
    template_name = "nonexistent_template"

    # Test and Assert
    with pytest.raises(FileNotFoundError):
        load(tmp_path, template_name)


# LLM-generated content at query #173
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


# LLM-generated content at query #174
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


# LLM-generated content at query #175
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


# LLM-generated content at query #176
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


# LLM-generated content at query #177
#--------------------------

```python
def test_dump():
    # Setup
    replay_dir = 'test_replay_dir'
    template_name = 'test_template'
    context = {'cookiecutter': {'key': 'value'}}
    file_name = get_file_name(replay_dir, template_name)

    # Test
    dump(replay_dir, template_name, context)

    # Assert
    assert os.path.exists(file_name)
    with open(file_name, encoding="utf-8") as infile:
        loaded_context = json.load(infile)
    assert loaded_context == context

    # Cleanup
    if os.path.exists(file_name):
        os.remove(file_name)
    if os.path.exists(replay_dir):
        os.rmdir(replay_dir)


# LLM-generated content at query #178
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    replay_dir = tmp_path
    replay_file = get_file_name(replay_dir, template_name)

    # Create a test file
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test
    loaded_context = load(replay_dir, template_name)

    # Assert
    assert loaded_context == context

    # Test with missing cookiecutter key
    invalid_context = {"key": "value"}
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(invalid_context, outfile, indent=2)

    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


# LLM-generated content at query #179
#--------------------------

```python
def test_dump():
    # Setup
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    expected_file_path = os.path.join(replay_dir, f"{template_name}.json")

    # Ensure the directory is clean before the test
    if os.path.exists(replay_dir):
        import shutil
        shutil.rmtree(replay_dir)

    # Execute
    dump(replay_dir, template_name, context)

    # Verify
    assert os.path.exists(expected_file_path)
    with open(expected_file_path, encoding="utf-8") as infile:
        loaded_context = json.load(infile)
    assert loaded_context == context

    # Cleanup
    if os.path.exists(replay_dir):
        import shutil
        shutil.rmtree(replay_dir)


# LLM-generated content at query #180
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


# LLM-generated content at query #181
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    replay_dir = tmp_path
    replay_file = get_file_name(replay_dir, template_name)

    # Create a test file
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test
    loaded_context = load(replay_dir, template_name)

    # Assert
    assert loaded_context == context


# LLM-generated content at query #182
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test_project"}}
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    replay_file = get_file_name(replay_dir, template_name)

    # Write test data
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test
    loaded_context = load(replay_dir, template_name)

    # Assert
    assert loaded_context == context
    assert loaded_context["cookiecutter"]["project_name"] == "test_project"

def test_load_missing_cookiecutter_key(tmp_path):
    # Setup
    template_name = "test_template"
    context = {"project_name": "test_project"}  # Missing 'cookiecutter' key
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    replay_file = get_file_name(replay_dir, template_name)

    # Write test data
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test and Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


# LLM-generated content at query #183
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test_project"}}

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
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    template_name = "test_template"
    context = {"project_name": "test_project"}  # Missing 'cookiecutter' key

    # Create a test file
    replay_file = get_file_name(replay_dir, template_name)
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test and Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)

def test_load_file_not_found(tmp_path):
    # Setup
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    template_name = "nonexistent_template"

    # Test and Assert
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


# LLM-generated content at query #184
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


# LLM-generated content at query #185
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

    # Test with .json suffix
    replay_file_json = get_file_name(replay_dir, f"{template_name}.json")
    with open(replay_file_json, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    loaded_context_json = load(replay_dir, f"{template_name}.json")
    assert loaded_context_json == context

    # Test error case: missing cookiecutter key
    invalid_context = {"project_name": "test_project"}
    invalid_replay_file = get_file_name(replay_dir, "invalid_template")
    with open(invalid_replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(invalid_context, outfile, indent=2)

    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, "invalid_template")


# LLM-generated content at query #186
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    context = {'cookiecutter': {'project_name': 'test_project'}}
    template_name = 'test_template'
    replay_dir = tmp_path / 'replay'
    replay_dir.mkdir()

    # Write test data
    replay_file = get_file_name(replay_dir, template_name)
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test
    loaded_context = load(replay_dir, template_name)

    # Assert
    assert loaded_context == context

    # Test with missing cookiecutter key
    bad_context = {'project_name': 'test_project'}
    replay_file_bad = get_file_name(replay_dir, 'bad_template')
    with open(replay_file_bad, 'w', encoding="utf-8") as outfile:
        json.dump(bad_context, outfile, indent=2)

    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, 'bad_template')


# LLM-generated content at query #187
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


# LLM-generated content at query #188
#--------------------------

```python
def test_dump():
    # Test successful dump
    test_dir = "test_replay_dir"
    test_template = "test_template"
    test_context = {"cookiecutter": {"key": "value"}}

    dump(test_dir, test_template, test_context)

    # Verify file was created with correct content
    replay_file = get_file_name(test_dir, test_template)
    assert os.path.exists(replay_file)

    with open(replay_file, encoding="utf-8") as f:
        loaded_context = json.load(f)

    assert loaded_context == test_context

    # Test with .json suffix
    test_template_json = "test_template.json"
    dump(test_dir, test_template_json, test_context)

    replay_file_json = get_file_name(test_dir, test_template_json)
    assert os.path.exists(replay_file_json)

    with open(replay_file_json, encoding="utf-8") as f:
        loaded_context_json = json.load(f)

    assert loaded_context_json == test_context

    # Test ValueError when cookiecutter key is missing
    invalid_context = {"key": "value"}
    with pytest.raises(ValueError) as excinfo:
        dump(test_dir, test_template, invalid_context)

    assert "Context is required to contain a cookiecutter key" in str(excinfo.value)

    # Clean up
    os.remove(replay_file)
    os.remove(replay_file_json)
    os.rmdir(test_dir)


# LLM-generated content at query #189
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


# LLM-generated content at query #190
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
    context = {"key": "value"}  # Missing 'cookiecutter' key
    replay_file = tmp_path / f"{template_name}.json"
    with open(replay_file, "w", encoding="utf-8") as f:
        json.dump(context, f, indent=2)

    # Test and Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(tmp_path, template_name)

def test_load_with_json_suffix(tmp_path):
    # Setup
    template_name = "test_template.json"
    context = {"cookiecutter": {"key": "value"}}
    replay_file = tmp_path / template_name
    with open(replay_file, "w", encoding="utf-8") as f:
        json.dump(context, f, indent=2)

    # Test
    result = load(tmp_path, template_name)

    # Assert
    assert result == context


# LLM-generated content at query #191
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    test_context = {'cookiecutter': {'project_name': 'test_project'}}
    template_name = 'test_template'
    replay_dir = tmp_path / 'replay'
    replay_dir.mkdir()

    # Create a test file
    replay_file = replay_dir / f'{template_name}.json'
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(test_context, f, indent=2)

    # Test
    result = load(replay_dir, template_name)

    # Assert
    assert result == test_context


# LLM-generated content at query #192
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


# LLM-generated content at query #193
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    test_context = {'cookiecutter': {'project_name': 'test'}}
    template_name = 'test_template'
    replay_dir = tmp_path
    replay_file = get_file_name(replay_dir, template_name)

    # Create a test file
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(test_context, outfile, indent=2)

    # Test
    result = load(replay_dir, template_name)

    # Assert
    assert result == test_context

def test_load_missing_cookiecutter_key(tmp_path):
    # Setup
    test_context = {'project_name': 'test'}
    template_name = 'test_template'
    replay_dir = tmp_path
    replay_file = get_file_name(replay_dir, template_name)

    # Create a test file without cookiecutter key
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(test_context, outfile, indent=2)

    # Test and Assert
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, template_name)


# LLM-generated content at query #194
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


# LLM-generated content at query #195
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


# LLM-generated content at query #196
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    test_context = {
        'cookiecutter': {
            'project_name': 'test_project',
            'project_slug': 'test_project_slug'
        }
    }
    template_name = 'test_template'
    replay_dir = tmp_path
    replay_file = get_file_name(replay_dir, template_name)

    # Create the test file
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(test_context, outfile, indent=2)

    # Test
    result = load(replay_dir, template_name)

    # Assert
    assert result == test_context


# LLM-generated content at query #197
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


# LLM-generated content at query #198
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


# LLM-generated content at query #199
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
    with open(expected_file, encoding="utf-8") as infile:
        loaded_context = json.load(infile)
    assert loaded_context == context

    # Cleanup
    os.remove(expected_file)
    os.rmdir(replay_dir)


# LLM-generated content at query #200
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


# LLM-generated content at query #201
#--------------------------

```python
def test_load(mocker, tmp_path):
    # Setup
    template_name = "test_template"
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()

    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "project_slug": "test_project"
        }
    }

    replay_file = get_file_name(replay_dir, template_name)
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Execute
    result = load(replay_dir, template_name)

    # Assert
    assert result == context

    # Test with missing cookiecutter key
    invalid_context = {"invalid_key": "value"}
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(invalid_context, outfile, indent=2)

    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)

    # Test with non-existent file
    non_existent_file = get_file_name(replay_dir, "non_existent")
    with pytest.raises(FileNotFoundError):
        load(replay_dir, "non_existent")


# LLM-generated content at query #202
#--------------------------

```python
def test_dump():
    # Setup
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    expected_file = os.path.join(replay_dir, "test_template.json")

    # Exercise
    dump(replay_dir, template_name, context)

    # Verify
    assert os.path.exists(expected_file)
    with open(expected_file, encoding="utf-8") as f:
        loaded_context = json.load(f)
    assert loaded_context == context

    # Cleanup
    os.remove(expected_file)
    os.rmdir(replay_dir)


# LLM-generated content at query #203
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    test_data = {"cookiecutter": {"project_name": "test"}}
    template_name = "test_template"
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()

    # Create a test file
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(test_data, f, indent=2)

    # Test
    result = load(replay_dir, template_name)

    # Assert
    assert result == test_data


# LLM-generated content at query #204
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
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
    context = {"key": "value"}  # Missing 'cookiecutter' key
    replay_file = tmp_path / f"{template_name}.json"

    # Create a test file
    with open(replay_file, "w", encoding="utf-8") as f:
        json.dump(context, f, indent=2)

    # Test and Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(tmp_path, template_name)

def test_load_file_not_found(tmp_path):
    # Setup
    template_name = "nonexistent_template"

    # Test and Assert
    with pytest.raises(FileNotFoundError):
        load(tmp_path, template_name)


# LLM-generated content at query #205
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    test_context = {"cookiecutter": {"project_name": "test_project"}}
    template_name = "test_template"
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()

    # Create a test file
    replay_file = get_file_name(replay_dir, template_name)
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(test_context, outfile, indent=2)

    # Test
    loaded_context = load(replay_dir, template_name)

    # Assert
    assert loaded_context == test_context

    # Test with .json suffix
    replay_file_json = get_file_name(replay_dir, f"{template_name}.json")
    with open(replay_file_json, 'w', encoding="utf-8") as outfile:
        json.dump(test_context, outfile, indent=2)

    loaded_context_json = load(replay_dir, f"{template_name}.json")
    assert loaded_context_json == test_context

    # Test missing cookiecutter key
    invalid_context = {"project_name": "test_project"}
    replay_file_invalid = get_file_name(replay_dir, "invalid_template")
    with open(replay_file_invalid, 'w', encoding="utf-8") as outfile:
        json.dump(invalid_context, outfile, indent=2)

    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, "invalid_template")


# LLM-generated content at query #206
#--------------------------

```python
def test_dump():
    # Test dump function
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}

    # Call the function
    dump(replay_dir, template_name, context)

    # Check if file is created
    file_name = get_file_name(replay_dir, template_name)
    assert os.path.exists(file_name)

    # Check if file content is correct
    with open(file_name, encoding="utf-8") as infile:
        loaded_context = json.load(infile)
    assert loaded_context == context

    # Clean up
    os.remove(file_name)
    os.rmdir(replay_dir)


# LLM-generated content at query #207
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    test_context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "test_author"
        }
    }
    template_name = "test_template"
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()

    # Create a test file
    replay_file = get_file_name(replay_dir, template_name)
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(test_context, f, indent=2)

    # Test
    result = load(replay_dir, template_name)

    # Assert
    assert result == test_context


# LLM-generated content at query #208
#--------------------------

```python
def test_dump():
    # Setup
    import tempfile
    import os
    from pathlib import Path

    replay_dir = tempfile.mkdtemp()
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}

    # Execute
    dump(replay_dir, template_name, context)

    # Assert
    replay_file = get_file_name(replay_dir, template_name)
    assert os.path.exists(replay_file)

    with open(replay_file, encoding="utf-8") as infile:
        loaded_context = json.load(infile)

    assert loaded_context == context

    # Cleanup
    os.remove(replay_file)
    os.rmdir(replay_dir)


# LLM-generated content at query #209
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


# LLM-generated content at query #210
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test_project"}}
    replay_file = get_file_name(replay_dir, template_name)

    # Create the replay directory and file
    replay_dir.mkdir()
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test
    loaded_context = load(replay_dir, template_name)

    # Assert
    assert loaded_context == context


# LLM-generated content at query #211
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    test_context = {'cookiecutter': {'project_name': 'test_project'}}
    template_name = 'test_template'
    replay_dir = tmp_path
    replay_file = get_file_name(replay_dir, template_name)

    # Create a test file
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(test_context, outfile, indent=2)

    # Test
    result = load(replay_dir, template_name)

    # Assert
    assert result == test_context


# LLM-generated content at query #212
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


# LLM-generated content at query #213
#--------------------------

```python
def test_load():
    # Setup
    replay_dir = "test_replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "project_slug": "test_project_slug"
        }
    }

    # Create the replay directory and dump the context
    os.makedirs(replay_dir, exist_ok=True)
    dump(replay_dir, template_name, context)

    # Test
    loaded_context = load(replay_dir, template_name)

    # Assert
    assert loaded_context == context

    # Cleanup
    os.remove(get_file_name(replay_dir, template_name))
    os.rmdir(replay_dir)


# LLM-generated content at query #214
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test_project"}}

    # Create replay directory and dump context
    dump(replay_dir, template_name, context)

    # Test
    loaded_context = load(replay_dir, template_name)

    # Assert
    assert loaded_context == context

def test_load_missing_cookiecutter_key(tmp_path):
    # Setup
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"project_name": "test_project"}  # Missing 'cookiecutter' key

    # Create replay directory and dump context
    dump(replay_dir, template_name, context)

    # Test and Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)

def test_load_nonexistent_file(tmp_path):
    # Setup
    replay_dir = tmp_path / "replay"
    template_name = "nonexistent_template"

    # Test and Assert
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_load():
    # Setup
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    os.makedirs(replay_dir, exist_ok=True)
    file_name = get_file_name(replay_dir, template_name)
    with open(file_name, "w", encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test
    result = load(replay_dir, template_name)

    # Assert
    assert result == context

    # Cleanup
    os.remove(file_name)
    os.rmdir(replay_dir)


# LLM-generated content at query #2
#--------------------------

```python
def test_dump():
    # Setup
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    expected_file_path = os.path.join(replay_dir, f"{template_name}.json")

    # Exercise
    dump(replay_dir, template_name, context)

    # Verify
    assert os.path.exists(expected_file_path)
    with open(expected_file_path, encoding="utf-8") as infile:
        loaded_context = json.load(infile)
    assert loaded_context == context

    # Cleanup
    os.remove(expected_file_path)
    os.rmdir(replay_dir)


# LLM-generated content at query #3
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


# LLM-generated content at query #4
#--------------------------

```python
def test_dump():
    # Test basic functionality
    replay_dir = "test_replay"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}

    dump(replay_dir, template_name, context)

    # Verify file was created
    expected_file = get_file_name(replay_dir, template_name)
    assert os.path.exists(expected_file)

    # Verify content
    with open(expected_file, encoding="utf-8") as f:
        loaded_context = json.load(f)
    assert loaded_context == context

    # Test with .json extension
    template_name_json = "test_template.json"
    dump(replay_dir, template_name_json, context)

    expected_file_json = get_file_name(replay_dir, template_name_json)
    assert os.path.exists(expected_file_json)

    # Test ValueError when cookiecutter key is missing
    invalid_context = {"key": "value"}
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, invalid_context)

    # Cleanup
    if os.path.exists(replay_dir):
        for file in os.listdir(replay_dir):
            os.remove(os.path.join(replay_dir, file))
        os.rmdir(replay_dir)


# LLM-generated content at query #5
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test_project"}}

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
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    template_name = "test_template"
    context = {"project_name": "test_project"}  # Missing 'cookiecutter' key

    # Create a test file
    replay_file = get_file_name(replay_dir, template_name)
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test and Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


# LLM-generated content at query #6
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    template_name = "test_template"
    replay_dir = tmp_path
    context = {"cookiecutter": {"project_name": "test_project"}}

    # Create a test file
    replay_file = get_file_name(replay_dir, template_name)
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test
    loaded_context = load(replay_dir, template_name)

    # Assert
    assert loaded_context == context

    # Test with missing cookiecutter key
    invalid_context = {"project_name": "test_project"}
    replay_file_invalid = get_file_name(replay_dir, "invalid_template")
    with open(replay_file_invalid, 'w', encoding="utf-8") as outfile:
        json.dump(invalid_context, outfile, indent=2)

    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, "invalid_template")


# LLM-generated content at query #7
#--------------------------

```python
def test_dump(tmp_path):
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "project_slug": "test_project_slug"
        }
    }

    dump(replay_dir, template_name, context)

    replay_file = get_file_name(replay_dir, template_name)
    assert replay_file.exists()

    with open(replay_file, encoding="utf-8") as infile:
        loaded_context = json.load(infile)

    assert loaded_context == context


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
def test_dump():
    # Setup
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    expected_file = os.path.join(replay_dir, "test_template.json")

    # Exercise
    dump(replay_dir, template_name, context)

    # Verify
    assert os.path.exists(expected_file)
    with open(expected_file, encoding="utf-8") as f:
        loaded_context = json.load(f)
    assert loaded_context == context

    # Cleanup
    os.remove(expected_file)
    os.rmdir(replay_dir)


# LLM-generated content at query #10
#--------------------------

```python
def test_dump(tmp_path):
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}

    dump(replay_dir, template_name, context)

    expected_file = replay_dir / f"{template_name}.json"
    assert expected_file.exists()

    with open(expected_file, encoding="utf-8") as f:
        loaded_context = json.load(f)

    assert loaded_context == context

def test_dump_with_json_suffix():
    replay_dir = "test_replay"
    template_name = "test_template.json"
    context = {"cookiecutter": {"key": "value"}}

    dump(replay_dir, template_name, context)

    expected_file = os.path.join(replay_dir, template_name)
    assert os.path.exists(expected_file)

    with open(expected_file, encoding="utf-8") as f:
        loaded_context = json.load(f)

    assert loaded_context == context

def test_dump_missing_cookiecutter_key():
    replay_dir = "test_replay"
    template_name = "test_template"
    context = {"key": "value"}

    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


# LLM-generated content at query #11
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


# LLM-generated content at query #12
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


# LLM-generated content at query #13
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


# LLM-generated content at query #14
#--------------------------

```python
def test_dump(tmp_path):
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "project_slug": "test_project_slug",
        }
    }

    dump(replay_dir, template_name, context)

    replay_file = get_file_name(replay_dir, template_name)
    assert os.path.exists(replay_file)

    with open(replay_file, encoding="utf-8") as infile:
        loaded_context = json.load(infile)

    assert loaded_context == context

def test_dump_missing_cookiecutter_key(tmp_path):
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "project_name": "test_project",
    }

    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


# LLM-generated content at query #15
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


# LLM-generated content at query #16
#--------------------------

```python
def test_dump(tmp_path):
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}

    dump(replay_dir, template_name, context)

    expected_file = replay_dir / f"{template_name}.json"
    assert expected_file.exists()

    with open(expected_file, encoding="utf-8") as f:
        loaded_context = json.load(f)

    assert loaded_context == context

def test_dump_with_json_suffix():
    replay_dir = "test_replay"
    template_name = "test_template.json"
    context = {"cookiecutter": {"key": "value"}}

    dump(replay_dir, template_name, context)

    expected_file = os.path.join(replay_dir, template_name)
    assert os.path.exists(expected_file)

    with open(expected_file, encoding="utf-8") as f:
        loaded_context = json.load(f)

    assert loaded_context == context

def test_dump_missing_cookiecutter_key():
    replay_dir = "test_replay"
    template_name = "test_template"
    context = {"key": "value"}

    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


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

def test_dump_with_json_suffix():
    replay_dir = "test_replay"
    template_name = "test_template.json"
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

def test_dump_without_cookiecutter_key():
    replay_dir = "test_replay"
    template_name = "test_template"
    context = {
        "project_name": "test_project",
        "author": "test_author"
    }

    with pytest.raises(ValueError) as excinfo:
        dump(replay_dir, template_name, context)

    assert "Context is required to contain a cookiecutter key" in str(excinfo.value)


# LLM-generated content at query #19
#--------------------------

```python
def test_dump():
    # Test successful dump
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    dump(replay_dir, template_name, context)

    # Verify file was created
    file_path = get_file_name(replay_dir, template_name)
    assert os.path.exists(file_path)

    # Verify content
    with open(file_path, encoding="utf-8") as f:
        loaded_context = json.load(f)
    assert loaded_context == context

    # Test with .json extension
    template_name_json = "test_template.json"
    dump(replay_dir, template_name_json, context)
    file_path_json = get_file_name(replay_dir, template_name_json)
    assert os.path.exists(file_path_json)

    # Test ValueError when cookiecutter key is missing
    invalid_context = {"key": "value"}
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, invalid_context)


# LLM-generated content at query #20
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


# LLM-generated content at query #21
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    replay_dir = tmp_path
    template_name = "test_template"
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
    replay_dir = tmp_path
    template_name = "test_template"
    context = {"key": "value"}  # Missing 'cookiecutter' key
    replay_file = get_file_name(replay_dir, template_name)

    # Create a test file
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test and Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


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


# LLM-generated content at query #24
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


# LLM-generated content at query #25
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


# LLM-generated content at query #26
#--------------------------

```python
def test_dump(tmp_path):
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test_project"}}

    dump(replay_dir, template_name, context)

    expected_file = replay_dir / f"{template_name}.json"
    assert expected_file.exists()

    with open(expected_file, encoding="utf-8") as f:
        loaded_context = json.load(f)

    assert loaded_context == context

def test_dump_with_json_suffix():
    replay_dir = "test_replay"
    template_name = "test_template.json"
    context = {"cookiecutter": {"project_name": "test_project"}}

    dump(replay_dir, template_name, context)

    expected_file = os.path.join(replay_dir, template_name)
    assert os.path.exists(expected_file)

    with open(expected_file, encoding="utf-8") as f:
        loaded_context = json.load(f)

    assert loaded_context == context

def test_dump_missing_cookiecutter_key():
    replay_dir = "test_replay"
    template_name = "test_template"
    context = {"project_name": "test_project"}

    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


# LLM-generated content at query #27
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

    expected_file = replay_dir / f"{template_name}.json"
    assert expected_file.exists()

    with open(expected_file, encoding="utf-8") as f:
        loaded_context = json.load(f)

    assert loaded_context == context

def test_dump_with_json_suffix():
    replay_dir = "test_replay"
    template_name = "test_template.json"
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "test_author"
        }
    }

    dump(replay_dir, template_name, context)

    expected_file = os.path.join(replay_dir, template_name)
    assert os.path.exists(expected_file)

    with open(expected_file, encoding="utf-8") as f:
        loaded_context = json.load(f)

    assert loaded_context == context

def test_dump_missing_cookiecutter_key():
    replay_dir = "test_replay"
    template_name = "test_template"
    context = {
        "project_name": "test_project",
        "author": "test_author"
    }

    with pytest.raises(ValueError) as excinfo:
        dump(replay_dir, template_name, context)

    assert "Context is required to contain a cookiecutter key" in str(excinfo.value)


# LLM-generated content at query #28
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


# LLM-generated content at query #29
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


# LLM-generated content at query #30
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
    replay_dir = tmp_path
    replay_file = get_file_name(replay_dir, template_name)

    # Write test data
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(test_context, outfile, indent=2)

    # Test
    result = load(replay_dir, template_name)

    # Assert
    assert result == test_context


# LLM-generated content at query #31
#--------------------------

```python
def test_dump():
    # Setup
    import tempfile
    from pathlib import Path
    test_dir = tempfile.mkdtemp()
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test_project"}}

    # Execute
    dump(test_dir, template_name, context)

    # Verify
    replay_file = get_file_name(test_dir, template_name)
    assert os.path.exists(replay_file)

    with open(replay_file, encoding="utf-8") as infile:
        loaded_context = json.load(infile)

    assert loaded_context == context

    # Cleanup
    os.remove(replay_file)
    os.rmdir(test_dir)


# LLM-generated content at query #32
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


# LLM-generated content at query #33
#--------------------------

```python
def test_dump(tmp_path):
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test"}}

    dump(replay_dir, template_name, context)

    expected_file = replay_dir / f"{template_name}.json"
    assert expected_file.exists()

    with open(expected_file, "r", encoding="utf-8") as f:
        loaded_context = json.load(f)

    assert loaded_context == context


# LLM-generated content at query #34
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


# LLM-generated content at query #35
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    replay_dir = tmp_path
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

    # Test with .json suffix
    template_name_json = "test_template.json"
    replay_file_json = get_file_name(replay_dir, template_name_json)

    with open(replay_file_json, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    loaded_context_json = load(replay_dir, template_name_json)
    assert loaded_context_json == context

    # Test error handling
    invalid_template_name = "invalid_template"
    invalid_replay_file = get_file_name(replay_dir, invalid_template_name)

    with open(invalid_replay_file, 'w', encoding="utf-8") as outfile:
        json.dump({"invalid_key": "value"}, outfile, indent=2)

    try:
        load(replay_dir, invalid_template_name)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "Context is required to contain a cookiecutter key"


# LLM-generated content at query #36
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    template_name = 'test_template'
    context = {'cookiecutter': {'project_name': 'test_project'}}
    replay_dir = str(tmp_path)
    replay_file = get_file_name(replay_dir, template_name)

    # Create the replay file
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test
    loaded_context = load(replay_dir, template_name)

    # Assert
    assert loaded_context == context

def test_load_missing_cookiecutter_key(tmp_path):
    # Setup
    template_name = 'test_template'
    context = {'project_name': 'test_project'}  # Missing 'cookiecutter' key
    replay_dir = str(tmp_path)
    replay_file = get_file_name(replay_dir, template_name)

    # Create the replay file
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test and Assert
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, template_name)

def test_load_json_file_extension(tmp_path):
    # Setup
    template_name = 'test_template.json'
    context = {'cookiecutter': {'project_name': 'test_project'}}
    replay_dir = str(tmp_path)
    replay_file = get_file_name(replay_dir, template_name)

    # Create the replay file
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test
    loaded_context = load(replay_dir, template_name)

    # Assert
    assert loaded_context == context


# LLM-generated content at query #37
#--------------------------

```python
def test_dump(tmp_path):
    # Test successful dump
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test_project"}}

    dump(replay_dir, template_name, context)

    # Verify file was created
    replay_file = get_file_name(replay_dir, template_name)
    assert os.path.exists(replay_file)

    # Verify file content
    with open(replay_file, encoding="utf-8") as infile:
        loaded_context = json.load(infile)
    assert loaded_context == context

    # Test with .json extension
    template_name_json = "test_template.json"
    dump(replay_dir, template_name_json, context)

    replay_file_json = get_file_name(replay_dir, template_name_json)
    assert os.path.exists(replay_file_json)

    # Test ValueError for missing cookiecutter key
    invalid_context = {"project_name": "test_project"}
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, invalid_context)


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

    # Exercise
    dump(replay_dir, template_name, context)

    # Verify
    assert os.path.exists(expected_file)
    with open(expected_file, encoding="utf-8") as f:
        loaded_context = json.load(f)
    assert loaded_context == context

    # Cleanup
    os.remove(expected_file)
    os.rmdir(replay_dir)


# LLM-generated content at query #40
#--------------------------

```python
def test_dump():
    # Test basic functionality
    replay_dir = "test_replay"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}

    dump(replay_dir, template_name, context)

    # Verify file was created
    expected_file = get_file_name(replay_dir, template_name)
    assert os.path.exists(expected_file)

    # Verify content
    with open(expected_file, encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded == context

    # Test with .json suffix
    template_name_json = "test_template.json"
    dump(replay_dir, template_name_json, context)

    expected_file_json = get_file_name(replay_dir, template_name_json)
    assert os.path.exists(expected_file_json)

    # Test error when no cookiecutter key
    bad_context = {"other_key": "value"}
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, bad_context)

    # Cleanup
    os.remove(expected_file)
    os.remove(expected_file_json)
    os.rmdir(replay_dir)


# LLM-generated content at query #41
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


# LLM-generated content at query #42
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


# LLM-generated content at query #43
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


# LLM-generated content at query #44
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

    # Test with missing cookiecutter key
    invalid_context = {"key": "value"}
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(invalid_context, outfile, indent=2)

    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


# LLM-generated content at query #45
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


# LLM-generated content at query #46
#--------------------------

```python
def test_load(tmp_path):
    # Create a test context with a cookiecutter key
    test_context = {
        'cookiecutter': {
            'project_name': 'test_project',
            'author': 'test_author'
        }
    }

    # Write the test context to a file
    test_file = tmp_path / 'test.json'
    with open(test_file, 'w', encoding="utf-8") as outfile:
        json.dump(test_context, outfile, indent=2)

    # Load the context using the function
    loaded_context = load(tmp_path, 'test')

    # Verify the loaded context matches the original
    assert loaded_context == test_context

    # Test with a file that doesn't have a cookiecutter key
    invalid_context = {'invalid_key': 'value'}
    invalid_file = tmp_path / 'invalid.json'
    with open(invalid_file, 'w', encoding="utf-8") as outfile:
        json.dump(invalid_context, outfile, indent=2)

    # Verify that loading an invalid context raises a ValueError
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(tmp_path, 'invalid')


# LLM-generated content at query #47
#--------------------------

```python
def test_dump(tmp_path):
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "project_slug": "test_project",
        }
    }

    dump(replay_dir, template_name, context)

    replay_file = get_file_name(replay_dir, template_name)
    assert os.path.exists(replay_file)

    with open(replay_file, encoding="utf-8") as infile:
        loaded_context = json.load(infile)

    assert loaded_context == context


# LLM-generated content at query #48
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


# LLM-generated content at query #49
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


# LLM-generated content at query #50
#--------------------------

```python
def test_load():
    # Setup
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    replay_file = get_file_name(replay_dir, template_name)
    os.makedirs(replay_dir, exist_ok=True)
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test
    loaded_context = load(replay_dir, template_name)

    # Assert
    assert loaded_context == context

    # Cleanup
    os.remove(replay_file)
    os.rmdir(replay_dir)


# LLM-generated content at query #51
#--------------------------

```python
def test_load():
    # Setup
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    replay_file = get_file_name(replay_dir, template_name)
    os.makedirs(replay_dir, exist_ok=True)
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test
    loaded_context = load(replay_dir, template_name)

    # Assert
    assert loaded_context == context

    # Cleanup
    os.remove(replay_file)
    os.rmdir(replay_dir)


# LLM-generated content at query #52
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    template_name = "test_template"
    replay_dir = tmp_path
    context = {"cookiecutter": {"project_name": "test_project"}}
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
    context = {"project_name": "test_project"}  # Missing 'cookiecutter' key
    replay_file = get_file_name(replay_dir, template_name)

    # Create a test file
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test and Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


# LLM-generated content at query #53
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


# LLM-generated content at query #54
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


# LLM-generated content at query #55
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


# LLM-generated content at query #56
#--------------------------

```python
def test_load():
    # Setup
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    file_name = get_file_name(replay_dir, template_name)
    os.makedirs(replay_dir, exist_ok=True)
    with open(file_name, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test
    loaded_context = load(replay_dir, template_name)

    # Assert
    assert loaded_context == context

    # Cleanup
    os.remove(file_name)
    os.rmdir(replay_dir)


# LLM-generated content at query #57
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


# LLM-generated content at query #58
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

    expected_file = replay_dir / f"{template_name}.json"
    assert expected_file.exists()

    with open(expected_file, "r", encoding="utf-8") as f:
        loaded_context = json.load(f)

    assert loaded_context == context

    # Test with .json suffix in template_name
    template_name_with_suffix = "test_template.json"
    dump(replay_dir, template_name_with_suffix, context)

    expected_file_with_suffix = replay_dir / template_name_with_suffix
    assert expected_file_with_suffix.exists()

    with open(expected_file_with_suffix, "r", encoding="utf-8") as f:
        loaded_context_with_suffix = json.load(f)

    assert loaded_context_with_suffix == context

    # Test ValueError when context doesn't contain cookiecutter key
    invalid_context = {"project_name": "test_project"}
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, invalid_context)


# LLM-generated content at query #59
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


# LLM-generated content at query #60
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


# LLM-generated content at query #61
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

    # Teardown
    os.remove(get_file_name(replay_dir, template_name))
    os.rmdir(replay_dir)


# LLM-generated content at query #62
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


# LLM-generated content at query #63
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


# LLM-generated content at query #64
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


# LLM-generated content at query #65
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

    # Test with .json suffix
    replay_file_json = get_file_name(replay_dir, f"{template_name}.json")
    with open(replay_file_json, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    loaded_context_json = load(replay_dir, f"{template_name}.json")
    assert loaded_context_json == context

    # Test error handling for missing cookiecutter key
    invalid_context = {"project_name": "test_project"}
    invalid_replay_file = tmp_path / "invalid_replay.json"
    with open(invalid_replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(invalid_context, outfile, indent=2)

    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(tmp_path, "invalid_replay")


# LLM-generated content at query #66
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


# LLM-generated content at query #67
#--------------------------

```python
def test_load():
    # Setup
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    replay_file = get_file_name(replay_dir, template_name)
    os.makedirs(replay_dir, exist_ok=True)
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test
    loaded_context = load(replay_dir, template_name)

    # Assert
    assert loaded_context == context

    # Cleanup
    os.remove(replay_file)
    os.rmdir(replay_dir)


# LLM-generated content at query #68
#--------------------------

```python
def test_dump(tmp_path):
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test_project"}}

    dump(replay_dir, template_name, context)

    expected_file = replay_dir / f"{template_name}.json"
    assert expected_file.exists()

    with open(expected_file, encoding="utf-8") as f:
        loaded_context = json.load(f)

    assert loaded_context == context

def test_dump_with_json_suffix():
    replay_dir = "test_replay"
    template_name = "test_template.json"
    context = {"cookiecutter": {"project_name": "test_project"}}

    dump(replay_dir, template_name, context)

    expected_file = os.path.join(replay_dir, template_name)
    assert os.path.exists(expected_file)

    with open(expected_file, encoding="utf-8") as f:
        loaded_context = json.load(f)

    assert loaded_context == context

def test_dump_missing_cookiecutter_key():
    replay_dir = "test_replay"
    template_name = "test_template"
    context = {"project_name": "test_project"}

    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


# LLM-generated content at query #69
#--------------------------

```python
def test_dump():
    # Test basic functionality
    replay_dir = 'test_replay_dir'
    template_name = 'test_template'
    context = {'cookiecutter': {'key': 'value'}}

    dump(replay_dir, template_name, context)

    # Verify file was created
    replay_file = get_file_name(replay_dir, template_name)
    assert os.path.exists(replay_file)

    # Verify content
    with open(replay_file, encoding="utf-8") as infile:
        loaded_context = json.load(infile)
    assert loaded_context == context

    # Test with .json suffix
    template_name_json = 'test_template.json'
    dump(replay_dir, template_name_json, context)

    replay_file_json = get_file_name(replay_dir, template_name_json)
    assert os.path.exists(replay_file_json)

    # Test ValueError when cookiecutter key is missing
    invalid_context = {'key': 'value'}
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        dump(replay_dir, template_name, invalid_context)


# LLM-generated content at query #70
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


# LLM-generated content at query #71
#--------------------------

```python
def test_dump(tmp_path):
    # Test basic functionality
    context = {'cookiecutter': {'project_name': 'test'}}
    template_name = 'test_template'
    replay_dir = str(tmp_path)

    dump(replay_dir, template_name, context)

    # Verify file was created
    expected_file = os.path.join(replay_dir, f'{template_name}.json')
    assert os.path.exists(expected_file)

    # Verify content
    with open(expected_file, encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded == context

    # Test with .json suffix in template_name
    template_name_json = 'test_template.json'
    dump(replay_dir, template_name_json, context)

    expected_file_json = os.path.join(replay_dir, template_name_json)
    assert os.path.exists(expected_file_json)

    # Test ValueError when cookiecutter key is missing
    bad_context = {'other_key': 'value'}
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        dump(replay_dir, template_name, bad_context)


# LLM-generated content at query #72
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


# LLM-generated content at query #73
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    replay_file = tmp_path / f"{template_name}.json"
    replay_file.write_text(json.dumps(context))

    # Test
    result = load(tmp_path, template_name)

    # Assert
    assert result == context

def test_load_without_json_extension(tmp_path):
    # Setup
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    replay_file = tmp_path / f"{template_name}.json"
    replay_file.write_text(json.dumps(context))

    # Test
    result = load(tmp_path, template_name)

    # Assert
    assert result == context

def test_load_missing_cookiecutter_key(tmp_path):
    # Setup
    template_name = "test_template"
    context = {"key": "value"}
    replay_file = tmp_path / f"{template_name}.json"
    replay_file.write_text(json.dumps(context))

    # Test and Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(tmp_path, template_name)

def test_load_nonexistent_file(tmp_path):
    # Setup
    template_name = "nonexistent_template"

    # Test and Assert
    with pytest.raises(FileNotFoundError):
        load(tmp_path, template_name)


# LLM-generated content at query #74
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


# LLM-generated content at query #75
#--------------------------

```python
def test_load():
    # Setup
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    replay_file = get_file_name(replay_dir, template_name)

    # Create test directory and file
    make_sure_path_exists(replay_dir)
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test
    loaded_context = load(replay_dir, template_name)

    # Assert
    assert loaded_context == context

    # Cleanup
    os.remove(replay_file)
    os.rmdir(replay_dir)


# LLM-generated content at query #76
#--------------------------

```python
def test_dump(tmp_path):
    # Test successful dump
    context = {'cookiecutter': {'project_name': 'test'}}
    dump(tmp_path, 'test_template', context)

    # Verify file was created
    expected_file = tmp_path / 'test_template.json'
    assert expected_file.exists()

    # Verify content
    with open(expected_file, 'r', encoding="utf-8") as f:
        loaded_context = json.load(f)
    assert loaded_context == context

    # Test with .json suffix
    dump(tmp_path, 'test_template.json', context)
    expected_file = tmp_path / 'test_template.json'
    assert expected_file.exists()

    # Test missing cookiecutter key
    invalid_context = {'project_name': 'test'}
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        dump(tmp_path, 'test_template', invalid_context)


# LLM-generated content at query #77
#--------------------------

```python
def test_dump():
    import tempfile
    import os
    from pathlib import Path

    # Setup
    replay_dir = tempfile.mkdtemp()
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test_project"}}

    # Execute
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


# LLM-generated content at query #78
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


# LLM-generated content at query #79
#--------------------------

```python
def test_load():
    # Setup
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    replay_file = get_file_name(replay_dir, template_name)
    make_sure_path_exists(replay_dir)
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test
    loaded_context = load(replay_dir, template_name)

    # Assert
    assert loaded_context == context

    # Cleanup
    os.remove(replay_file)
    os.rmdir(replay_dir)


# LLM-generated content at query #80
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


# LLM-generated content at query #81
#--------------------------

```python
def test_dump(tmp_path):
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test_project"}}

    dump(replay_dir, template_name, context)

    expected_file = replay_dir / f"{template_name}.json"
    assert expected_file.exists()

    with open(expected_file, encoding="utf-8") as f:
        loaded_context = json.load(f)

    assert loaded_context == context

def test_dump_with_json_suffix():
    replay_dir = "test_replay"
    template_name = "test_template.json"
    context = {"cookiecutter": {"project_name": "test_project"}}

    dump(replay_dir, template_name, context)

    expected_file = os.path.join(replay_dir, template_name)
    assert os.path.exists(expected_file)

    with open(expected_file, encoding="utf-8") as f:
        loaded_context = json.load(f)

    assert loaded_context == context

def test_dump_missing_cookiecutter_key():
    replay_dir = "test_replay"
    template_name = "test_template"
    context = {"project_name": "test_project"}

    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


# LLM-generated content at query #82
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


# LLM-generated content at query #83
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

    expected_file = replay_dir / f"{template_name}.json"
    assert expected_file.exists()

    with open(expected_file, encoding="utf-8") as f:
        loaded_context = json.load(f)

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


# LLM-generated content at query #84
#--------------------------

```python
def test_dump():
    # Setup
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    expected_file = os.path.join(replay_dir, "test_template.json")

    # Exercise
    dump(replay_dir, template_name, context)

    # Verify
    assert os.path.exists(expected_file)
    with open(expected_file, encoding="utf-8") as f:
        loaded_context = json.load(f)
    assert loaded_context == context

    # Cleanup
    os.remove(expected_file)
    os.rmdir(replay_dir)


# LLM-generated content at query #85
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    test_context = {'cookiecutter': {'project_name': 'test_project'}}
    template_name = 'test_template'
    replay_dir = tmp_path
    replay_file = get_file_name(replay_dir, template_name)

    # Create a test file
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(test_context, outfile, indent=2)

    # Test
    result = load(replay_dir, template_name)

    # Assert
    assert result == test_context

    # Test with .json suffix
    template_name_with_suffix = 'test_template.json'
    replay_file_with_suffix = get_file_name(replay_dir, template_name_with_suffix)

    with open(replay_file_with_suffix, 'w', encoding="utf-8") as outfile:
        json.dump(test_context, outfile, indent=2)

    result_with_suffix = load(replay_dir, template_name_with_suffix)
    assert result_with_suffix == test_context

    # Test error case
    invalid_context = {'invalid_key': 'value'}
    invalid_replay_file = get_file_name(replay_dir, 'invalid_template')

    with open(invalid_replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(invalid_context, outfile, indent=2)

    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, 'invalid_template')


# LLM-generated content at query #86
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
    replay_dir = tmp_path
    replay_file = get_file_name(replay_dir, template_name)

    # Create a test file
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(test_context, outfile, indent=2)

    # Test
    result = load(replay_dir, template_name)

    # Assert
    assert result == test_context

    # Test with missing cookiecutter key
    invalid_context = {'invalid_key': 'value'}
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(invalid_context, outfile, indent=2)

    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, template_name)


# LLM-generated content at query #87
#--------------------------

```python
def test_dump():
    # Setup
    import tempfile
    import os
    from pathlib import Path

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
    replay_file = get_file_name(replay_dir, template_name)
    assert os.path.exists(replay_file)

    with open(replay_file, encoding="utf-8") as infile:
        loaded_context = json.load(infile)

    assert loaded_context == context

    # Cleanup
    os.remove(replay_file)
    os.rmdir(replay_dir)


# LLM-generated content at query #88
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


# LLM-generated content at query #89
#--------------------------

```python
def test_dump():
    # Setup
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    expected_file = os.path.join(replay_dir, f"{template_name}.json")

    # Exercise
    dump(replay_dir, template_name, context)

    # Verify
    assert os.path.exists(expected_file)
    with open(expected_file, encoding="utf-8") as f:
        loaded_context = json.load(f)
    assert loaded_context == context

    # Cleanup
    os.remove(expected_file)
    os.rmdir(replay_dir)


# LLM-generated content at query #90
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

    expected_file = replay_dir / f"{template_name}.json"
    assert expected_file.exists()

    with open(expected_file, encoding="utf-8") as f:
        loaded_context = json.load(f)

    assert loaded_context == context

def test_dump_with_json_suffix():
    replay_dir = "test_replay"
    template_name = "test_template.json"
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "test_author"
        }
    }

    dump(replay_dir, template_name, context)

    expected_file = os.path.join(replay_dir, template_name)
    assert os.path.exists(expected_file)

    with open(expected_file, encoding="utf-8") as f:
        loaded_context = json.load(f)

    assert loaded_context == context

def test_dump_missing_cookiecutter_key():
    replay_dir = "test_replay"
    template_name = "test_template"
    context = {
        "project_name": "test_project",
        "author": "test_author"
    }

    with pytest.raises(ValueError) as excinfo:
        dump(replay_dir, template_name, context)

    assert "Context is required to contain a cookiecutter key" in str(excinfo.value)


# LLM-generated content at query #91
#--------------------------

```python
def test_dump(tmp_path):
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}

    dump(replay_dir, template_name, context)

    expected_file = replay_dir / f"{template_name}.json"
    assert expected_file.exists()

    with open(expected_file, encoding="utf-8") as f:
        loaded_context = json.load(f)

    assert loaded_context == context

def test_dump_with_json_suffix():
    replay_dir = "test_replay"
    template_name = "test_template.json"
    context = {"cookiecutter": {"key": "value"}}

    dump(replay_dir, template_name, context)

    expected_file = os.path.join(replay_dir, template_name)
    assert os.path.exists(expected_file)

    with open(expected_file, encoding="utf-8") as f:
        loaded_context = json.load(f)

    assert loaded_context == context

def test_dump_missing_cookiecutter_key():
    replay_dir = "test_replay"
    template_name = "test_template"
    context = {"key": "value"}

    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


# LLM-generated content at query #92
#--------------------------

```python
def test_dump(tmp_path):
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}

    dump(replay_dir, template_name, context)

    expected_file = replay_dir / f"{template_name}.json"
    assert expected_file.exists()

    with open(expected_file, encoding="utf-8") as f:
        loaded_context = json.load(f)

    assert loaded_context == context

def test_dump_with_json_suffix():
    replay_dir = "test_replay"
    template_name = "test_template.json"
    context = {"cookiecutter": {"key": "value"}}

    dump(replay_dir, template_name, context)

    expected_file = os.path.join(replay_dir, template_name)
    assert os.path.exists(expected_file)

    with open(expected_file, encoding="utf-8") as f:
        loaded_context = json.load(f)

    assert loaded_context == context

def test_dump_missing_cookiecutter_key():
    replay_dir = "test_replay"
    template_name = "test_template"
    context = {"key": "value"}

    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


# LLM-generated content at query #93
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


# LLM-generated content at query #94
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

    # Test with missing cookiecutter key
    invalid_context = {'invalid_key': 'value'}
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(invalid_context, outfile, indent=2)

    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(tmp_path, test_template_name)


# LLM-generated content at query #95
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


# LLM-generated content at query #96
#--------------------------

```python
def test_dump(tmp_path):
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test_project"}}

    dump(replay_dir, template_name, context)

    expected_file = replay_dir / f"{template_name}.json"
    assert expected_file.exists()

    with open(expected_file, encoding="utf-8") as f:
        loaded_context = json.load(f)

    assert loaded_context == context


# LLM-generated content at query #97
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


# LLM-generated content at query #98
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    test_context = {'cookiecutter': {'project_name': 'test_project'}}
    template_name = 'test_template'
    replay_dir = tmp_path / 'replay'
    replay_dir.mkdir()

    # Create a test file
    replay_file = get_file_name(replay_dir, template_name)
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(test_context, f, indent=2)

    # Test
    result = load(replay_dir, template_name)

    # Assert
    assert result == test_context


# LLM-generated content at query #99
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


# LLM-generated content at query #100
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test_project"}}

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
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    template_name = "test_template"
    context = {"project_name": "test_project"}  # Missing 'cookiecutter' key

    # Create a test file
    replay_file = get_file_name(replay_dir, template_name)
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test and Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)

def test_load_with_json_suffix(tmp_path):
    # Setup
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    template_name = "test_template.json"
    context = {"cookiecutter": {"project_name": "test_project"}}

    # Create a test file
    replay_file = get_file_name(replay_dir, template_name)
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test
    loaded_context = load(replay_dir, template_name)

    # Assert
    assert loaded_context == context


# LLM-generated content at query #101
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


# LLM-generated content at query #102
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


# LLM-generated content at query #103
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

def test_load_file_not_found(tmp_path):
    # Setup
    template_name = "nonexistent_template"
    replay_dir = tmp_path

    # Test and Assert
    with pytest.raises(FileNotFoundError):
        load(replay_dir, template_name)


# LLM-generated content at query #104
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


# LLM-generated content at query #105
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


# LLM-generated content at query #106
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test_project"}}
    replay_file = tmp_path / "test_template.json"

    # Create a test file
    with open(replay_file, "w", encoding="utf-8") as f:
        json.dump(context, f, indent=2)

    # Test
    result = load(tmp_path, template_name)

    # Assert
    assert result == context


# LLM-generated content at query #107
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
    replay_dir = "test_replay"
    template_name = "test_template.json"
    context = {"cookiecutter": {"project_name": "test_project"}}

    dump(replay_dir, template_name, context)

    expected_file = os.path.join(replay_dir, template_name)
    assert os.path.exists(expected_file)

    with open(expected_file, "r", encoding="utf-8") as f:
        loaded_context = json.load(f)

    assert loaded_context == context

def test_dump_missing_cookiecutter_key():
    replay_dir = "test_replay"
    template_name = "test_template"
    context = {"project_name": "test_project"}

    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


# LLM-generated content at query #108
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test_project"}}
    replay_file = tmp_path / "test_template.json"

    # Create a test file
    with open(replay_file, "w", encoding="utf-8") as f:
        json.dump(context, f, indent=2)

    # Test
    result = load(tmp_path, template_name)

    # Assert
    assert result == context


# LLM-generated content at query #109
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    replay_file = tmp_path / f"{template_name}.json"

    # Create a test file
    with open(replay_file, "w", encoding="utf-8") as f:
        json.dump(context, f, indent=2)

    # Test
    result = load(tmp_path, template_name)

    # Assert
    assert result == context

def test_load_without_json_extension(tmp_path):
    # Setup
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
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
    context = {"key": "value"}
    replay_file = tmp_path / f"{template_name}.json"

    # Create a test file
    with open(replay_file, "w", encoding="utf-8") as f:
        json.dump(context, f, indent=2)

    # Test and Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(tmp_path, template_name)

def test_load_file_not_found(tmp_path):
    # Setup
    template_name = "non_existent_template"

    # Test and Assert
    with pytest.raises(FileNotFoundError):
        load(tmp_path, template_name)


# LLM-generated content at query #110
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
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(test_context, outfile, indent=2)

    # Test
    loaded_context = load(replay_dir, template_name)

    # Assert
    assert loaded_context == test_context


# LLM-generated content at query #111
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


# LLM-generated content at query #112
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


# LLM-generated content at query #113
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    test_context = {
        'cookiecutter': {
            'project_name': 'test_project',
            'project_slug': 'test_project'
        }
    }
    template_name = 'test_template'
    replay_dir = tmp_path
    replay_file = get_file_name(replay_dir, template_name)

    # Create a test file
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(test_context, outfile, indent=2)

    # Test
    result = load(replay_dir, template_name)

    # Assert
    assert result == test_context


# LLM-generated content at query #114
#--------------------------

```python
def test_dump():
    # Setup
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    expected_file = os.path.join(replay_dir, "test_template.json")

    # Exercise
    dump(replay_dir, template_name, context)

    # Verify
    assert os.path.exists(expected_file)
    with open(expected_file, encoding="utf-8") as infile:
        loaded_context = json.load(infile)
    assert loaded_context == context

    # Cleanup
    if os.path.exists(replay_dir):
        os.remove(expected_file)
        os.rmdir(replay_dir)


# LLM-generated content at query #115
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


# LLM-generated content at query #116
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
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
    context = {"key": "value"}  # Missing 'cookiecutter' key
    replay_file = tmp_path / f"{template_name}.json"

    # Write test data
    with open(replay_file, "w", encoding="utf-8") as f:
        json.dump(context, f, indent=2)

    # Test & Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(tmp_path, template_name)

def test_load_with_json_suffix(tmp_path):
    # Setup
    template_name = "test_template.json"
    context = {"cookiecutter": {"key": "value"}}
    replay_file = tmp_path / template_name

    # Write test data
    with open(replay_file, "w", encoding="utf-8") as f:
        json.dump(context, f, indent=2)

    # Test
    result = load(tmp_path, template_name)

    # Assert
    assert result == context


# LLM-generated content at query #117
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test_project"}}

    # Create a test file
    replay_file = get_file_name(replay_dir, template_name)
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test
    loaded_context = load(replay_dir, template_name)

    # Assert
    assert loaded_context == context


# LLM-generated content at query #118
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()

    # Write test data
    replay_file = get_file_name(replay_dir, template_name)
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test
    loaded_context = load(replay_dir, template_name)

    # Assert
    assert loaded_context == context

    # Test with missing cookiecutter key
    bad_context = {"key": "value"}
    bad_replay_file = get_file_name(replay_dir, "bad_template")
    with open(bad_replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(bad_context, outfile, indent=2)

    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, "bad_template")


# LLM-generated content at query #119
#--------------------------

```python
def test_dump():
    # Setup
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    expected_file = os.path.join(replay_dir, f"{template_name}.json")

    # Exercise
    dump(replay_dir, template_name, context)

    # Verify
    assert os.path.exists(expected_file)
    with open(expected_file, encoding="utf-8") as infile:
        loaded_context = json.load(infile)
    assert loaded_context == context

    # Teardown
    if os.path.exists(replay_dir):
        os.remove(expected_file)
        os.rmdir(replay_dir)


# LLM-generated content at query #120
#--------------------------

```python
def test_dump():
    # Test that dump creates a file with the correct content
    import tempfile
    import os
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "test_template"
        context = {"cookiecutter": {"project_name": "test_project"}}

        dump(replay_dir, template_name, context)

        expected_file = get_file_name(replay_dir, template_name)
        assert os.path.exists(expected_file)

        with open(expected_file, encoding="utf-8") as f:
            loaded_context = json.load(f)

        assert loaded_context == context

    # Test that dump raises ValueError if context does not contain 'cookiecutter' key
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "test_template"
        context = {"project_name": "test_project"}

        with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
            dump(replay_dir, template_name, context)


# LLM-generated content at query #121
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    replay_dir = tmp_path
    replay_file = get_file_name(replay_dir, template_name)

    # Create a test file
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test
    loaded_context = load(replay_dir, template_name)

    # Assert
    assert loaded_context == context


# LLM-generated content at query #122
#--------------------------

```python
def test_dump():
    # Setup
    replay_dir = "test_replay"
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


# LLM-generated content at query #123
#--------------------------

```python
def test_dump():
    # Test dump function
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}

    # Call the dump function
    dump(replay_dir, template_name, context)

    # Check if the file was created
    file_path = get_file_name(replay_dir, template_name)
    assert os.path.exists(file_path)

    # Check if the file content is correct
    with open(file_path, encoding="utf-8") as infile:
        loaded_context = json.load(infile)
    assert loaded_context == context

    # Test with template_name ending with .json
    template_name_json = "test_template.json"
    dump(replay_dir, template_name_json, context)
    file_path_json = get_file_name(replay_dir, template_name_json)
    assert os.path.exists(file_path_json)

    # Test with missing cookiecutter key
    invalid_context = {"key": "value"}
    try:
        dump(replay_dir, template_name, invalid_context)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "Context is required to contain a cookiecutter key"


# LLM-generated content at query #124
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
    os.remove(file_name)
    os.rmdir(replay_dir)


# LLM-generated content at query #125
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


# LLM-generated content at query #126
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


# LLM-generated content at query #127
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    test_context = {
        'cookiecutter': {
            'project_name': 'test_project',
            'author': 'test_author'
        },
        'other_data': 'value'
    }
    template_name = 'test_template'
    replay_file = tmp_path / f'{template_name}.json'

    # Create a test file
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(test_context, f, indent=2)

    # Test
    result = load(tmp_path, template_name)

    # Assert
    assert result == test_context


# LLM-generated content at query #128
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


# LLM-generated content at query #129
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test_project"}}
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()

    # Create a test file
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, "w", encoding="utf-8") as f:
        json.dump(context, f, indent=2)

    # Test
    loaded_context = load(replay_dir, template_name)

    # Assert
    assert loaded_context == context


# LLM-generated content at query #130
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


# LLM-generated content at query #131
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

    expected_file = replay_dir / f"{template_name}.json"
    assert expected_file.exists()

    with open(expected_file, encoding="utf-8") as f:
        loaded_context = json.load(f)

    assert loaded_context == context

def test_dump_with_json_suffix():
    replay_dir = "test_replay"
    template_name = "test_template.json"
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "test_author"
        }
    }

    dump(replay_dir, template_name, context)

    expected_file = os.path.join(replay_dir, template_name)
    assert os.path.exists(expected_file)

    with open(expected_file, encoding="utf-8") as f:
        loaded_context = json.load(f)

    assert loaded_context == context

def test_dump_missing_cookiecutter_key():
    replay_dir = "test_replay"
    template_name = "test_template"
    context = {
        "project_name": "test_project",
        "author": "test_author"
    }

    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


# LLM-generated content at query #132
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


# LLM-generated content at query #133
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


# LLM-generated content at query #134
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    test_context = {'cookiecutter': {'project_name': 'test_project'}}
    template_name = 'test_template'
    replay_dir = tmp_path / 'replay'
    replay_dir.mkdir()

    # Create a test file
    replay_file = get_file_name(replay_dir, template_name)
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(test_context, f, indent=2)

    # Test
    result = load(replay_dir, template_name)

    # Assert
    assert result == test_context


# LLM-generated content at query #135
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


# LLM-generated content at query #136
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
    if os.path.exists(replay_file):
        os.remove(replay_file)
    if os.path.exists(replay_dir):
        os.rmdir(replay_dir)


# LLM-generated content at query #137
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


# LLM-generated content at query #138
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "test_author"
        }
    }
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
    context = {
        "project_name": "test_project",
        "author": "test_author"
    }
    replay_dir = tmp_path
    replay_file = get_file_name(replay_dir, template_name)

    # Create a test file without cookiecutter key
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test and Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


# LLM-generated content at query #139
#--------------------------

```python
def test_load():
    # Setup
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    replay_file = get_file_name(replay_dir, template_name)
    os.makedirs(replay_dir, exist_ok=True)
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test
    loaded_context = load(replay_dir, template_name)

    # Assert
    assert loaded_context == context

    # Teardown
    os.remove(replay_file)
    os.rmdir(replay_dir)


# LLM-generated content at query #140
#--------------------------

```python
def test_dump(tmp_path):
    # Test basic functionality
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test_project"}}

    dump(replay_dir, template_name, context)

    # Verify file was created
    expected_file = replay_dir / f"{template_name}.json"
    assert expected_file.exists()

    # Verify content
    with open(expected_file, encoding="utf-8") as f:
        loaded_context = json.load(f)
    assert loaded_context == context

    # Test with .json suffix in template_name
    template_name_with_suffix = "test_template.json"
    dump(replay_dir, template_name_with_suffix, context)

    expected_file_with_suffix = replay_dir / template_name_with_suffix
    assert expected_file_with_suffix.exists()

    # Test ValueError when cookiecutter key is missing
    invalid_context = {"project_name": "test_project"}
    with pytest.raises(ValueError) as exc_info:
        dump(replay_dir, template_name, invalid_context)
    assert "Context is required to contain a cookiecutter key" in str(exc_info.value)


# LLM-generated content at query #141
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


# LLM-generated content at query #142
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


# LLM-generated content at query #143
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


# LLM-generated content at query #144
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

    # Test directory creation
    new_dir = tmp_path / 'new_dir'
    dump(new_dir, 'test', context)
    assert os.path.exists(new_dir / 'test.json')

    # Test ValueError for missing cookiecutter key
    with pytest.raises(ValueError):
        dump(tmp_path, 'test', {'invalid': 'context'})


# LLM-generated content at query #145
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

    # Execute
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


# LLM-generated content at query #146
#--------------------------

```python
def test_dump():
    # Setup
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    expected_file = os.path.join(replay_dir, f"{template_name}.json")

    # Exercise
    dump(replay_dir, template_name, context)

    # Verify
    assert os.path.exists(expected_file)
    with open(expected_file, encoding="utf-8") as infile:
        loaded_context = json.load(infile)
    assert loaded_context == context

    # Cleanup
    os.remove(expected_file)
    os.rmdir(replay_dir)


# LLM-generated content at query #147
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test_project"}}
    replay_file = tmp_path / f"{template_name}.json"

    # Create a test file
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test
    loaded_context = load(tmp_path, template_name)

    # Assert
    assert loaded_context == context


# LLM-generated content at query #148
#--------------------------

```python
def test_dump():
    # Test basic functionality
    replay_dir = "test_replay"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}

    dump(replay_dir, template_name, context)

    # Verify file was created with correct content
    replay_file = get_file_name(replay_dir, template_name)
    assert os.path.exists(replay_file)

    with open(replay_file, encoding="utf-8") as infile:
        loaded_context = json.load(infile)

    assert loaded_context == context

    # Test with .json suffix in template name
    template_name_json = "test_template.json"
    dump(replay_dir, template_name_json, context)

    replay_file_json = get_file_name(replay_dir, template_name_json)
    assert os.path.exists(replay_file_json)

    with open(replay_file_json, encoding="utf-8") as infile:
        loaded_context_json = json.load(infile)

    assert loaded_context_json == context

    # Test ValueError when cookiecutter key is missing
    invalid_context = {"key": "value"}
    try:
        dump(replay_dir, template_name, invalid_context)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "Context is required to contain a cookiecutter key"

    # Clean up
    os.remove(replay_file)
    os.remove(replay_file_json)
    os.rmdir(replay_dir)


# LLM-generated content at query #149
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


# LLM-generated content at query #150
#--------------------------

```python
def test_dump():
    # Test basic functionality
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    dump(replay_dir, template_name, context)

    # Verify file was created
    file_path = get_file_name(replay_dir, template_name)
    assert os.path.exists(file_path)

    # Verify content
    with open(file_path, encoding="utf-8") as f:
        loaded_context = json.load(f)
    assert loaded_context == context

    # Test with .json extension
    template_name_json = "test_template.json"
    dump(replay_dir, template_name_json, context)
    file_path_json = get_file_name(replay_dir, template_name_json)
    assert os.path.exists(file_path_json)

    # Test error when cookiecutter key is missing
    with pytest.raises(ValueError):
        dump(replay_dir, template_name, {"other_key": "value"})

    # Clean up
    if os.path.exists(replay_dir):
        for file in os.listdir(replay_dir):
            os.remove(os.path.join(replay_dir, file))
        os.rmdir(replay_dir)


# LLM-generated content at query #151
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


# LLM-generated content at query #152
#--------------------------

```python
def test_dump(tmp_path):
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test_project"}}

    dump(replay_dir, template_name, context)

    expected_file = replay_dir / f"{template_name}.json"
    assert expected_file.exists()

    with open(expected_file, encoding="utf-8") as f:
        loaded_context = json.load(f)

    assert loaded_context == context


# LLM-generated content at query #153
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

def test_load_missing_cookiecutter_key(tmp_path):
    # Setup
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    template_name = "test_template"
    context = {"project_name": "test_project"}  # Missing 'cookiecutter' key
    replay_file = get_file_name(replay_dir, template_name)

    # Write test data
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test and Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)

def test_load_with_json_suffix(tmp_path):
    # Setup
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    template_name = "test_template.json"
    context = {"cookiecutter": {"project_name": "test_project"}}
    replay_file = get_file_name(replay_dir, template_name)

    # Write test data
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test
    loaded_context = load(replay_dir, template_name)

    # Assert
    assert loaded_context == context


# LLM-generated content at query #154
#--------------------------

```python
def test_dump():
    # Test that dump writes the correct context to a file
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}

    dump(replay_dir, template_name, context)

    # Verify the file was created and contains the correct data
    replay_file = get_file_name(replay_dir, template_name)
    assert os.path.exists(replay_file)

    with open(replay_file, encoding="utf-8") as infile:
        loaded_context = json.load(infile)

    assert loaded_context == context

    # Test that dump raises ValueError if context does not contain 'cookiecutter' key
    invalid_context = {"key": "value"}
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, invalid_context)


# LLM-generated content at query #155
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
    replay_file = get_file_name(replay_dir, template_name)

    # Write test data
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(test_context, outfile, indent=2)

    # Test
    result = load(replay_dir, template_name)

    # Assert
    assert result == test_context


# LLM-generated content at query #156
#--------------------------

```python
def test_dump(tmp_path):
    # Test successful dump
    context = {'cookiecutter': {'project_name': 'test'}}
    template_name = 'test_template'
    replay_dir = tmp_path / 'replay'

    dump(replay_dir, template_name, context)

    expected_file = replay_dir / f'{template_name}.json'
    assert expected_file.exists()

    with open(expected_file, encoding="utf-8") as f:
        loaded_context = json.load(f)

    assert loaded_context == context

    # Test dump with .json extension in template name
    template_name_json = 'test_template.json'
    dump(replay_dir, template_name_json, context)

    expected_file_json = replay_dir / template_name_json
    assert expected_file_json.exists()

    with open(expected_file_json, encoding="utf-8") as f:
        loaded_context = json.load(f)

    assert loaded_context == context

    # Test dump without cookiecutter key
    invalid_context = {'project_name': 'test'}

    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        dump(replay_dir, template_name, invalid_context)


# LLM-generated content at query #157
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
            "project_slug": "test_project_slug"
        }
    }
    replay_file = get_file_name(replay_dir, template_name)
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test
    loaded_context = load(replay_dir, template_name)

    # Assert
    assert loaded_context == context


# LLM-generated content at query #158
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


# LLM-generated content at query #159
#--------------------------

```python
def test_load():
    # Setup
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    replay_file = get_file_name(replay_dir, template_name)
    os.makedirs(replay_dir, exist_ok=True)
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test
    loaded_context = load(replay_dir, template_name)

    # Assert
    assert loaded_context == context

    # Cleanup
    os.remove(replay_file)
    os.rmdir(replay_dir)


# LLM-generated content at query #160
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


# LLM-generated content at query #161
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    replay_file = tmp_path / f"{template_name}.json"

    # Create a test file
    with open(replay_file, "w", encoding="utf-8") as f:
        json.dump(context, f, indent=2)

    # Test
    result = load(tmp_path, template_name)

    # Assert
    assert result == context

def test_load_without_json_extension(tmp_path):
    # Setup
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
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
    context = {"key": "value"}
    replay_file = tmp_path / f"{template_name}.json"

    # Create a test file
    with open(replay_file, "w", encoding="utf-8") as f:
        json.dump(context, f, indent=2)

    # Test and Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(tmp_path, template_name)

def test_load_file_not_found(tmp_path):
    # Setup
    template_name = "non_existent_template"

    # Test and Assert
    with pytest.raises(FileNotFoundError):
        load(tmp_path, template_name)


# LLM-generated content at query #162
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
    with open(replay_file, encoding="utf-8") as infile:
        loaded_context = json.load(infile)

    assert loaded_context == context


# LLM-generated content at query #163
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


# LLM-generated content at query #164
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    test_context = {
        'cookiecutter': {
            'project_name': 'test_project',
            'project_slug': 'test_project_slug'
        }
    }
    template_name = 'test_template'
    replay_dir = tmp_path / 'replay'
    replay_dir.mkdir()

    # Write test data
    replay_file = get_file_name(replay_dir, template_name)
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(test_context, outfile, indent=2)

    # Test
    result = load(replay_dir, template_name)

    # Assert
    assert result == test_context


# LLM-generated content at query #165
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


# LLM-generated content at query #166
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


# LLM-generated content at query #167
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


# LLM-generated content at query #168
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


# LLM-generated content at query #169
#--------------------------

```python
def test_load():
    # Setup
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    os.makedirs(replay_dir, exist_ok=True)
    dump(replay_dir, template_name, context)

    # Exercise
    loaded_context = load(replay_dir, template_name)

    # Verify
    assert loaded_context == context

    # Teardown
    os.remove(get_file_name(replay_dir, template_name))
    os.rmdir(replay_dir)


# LLM-generated content at query #170
#--------------------------

```python
def test_load():
    # Setup
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    replay_file = get_file_name(replay_dir, template_name)
    os.makedirs(replay_dir, exist_ok=True)
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test
    loaded_context = load(replay_dir, template_name)

    # Assert
    assert loaded_context == context

    # Cleanup
    os.remove(replay_file)
    os.rmdir(replay_dir)


# LLM-generated content at query #171
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


# LLM-generated content at query #172
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

    expected_file = replay_dir / f"{template_name}.json"
    assert expected_file.exists()

    with open(expected_file, encoding="utf-8") as f:
        loaded_context = json.load(f)

    assert loaded_context == context

def test_dump_with_json_suffix():
    replay_dir = "test_replay_dir"
    template_name = "test_template.json"
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "test_author"
        }
    }

    dump(replay_dir, template_name, context)

    expected_file = os.path.join(replay_dir, template_name)
    assert os.path.exists(expected_file)

    with open(expected_file, encoding="utf-8") as f:
        loaded_context = json.load(f)

    assert loaded_context == context

def test_dump_missing_cookiecutter_key():
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    context = {
        "project_name": "test_project",
        "author": "test_author"
    }

    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


# LLM-generated content at query #173
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test_project"}}

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
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    template_name = "test_template"
    context = {"project_name": "test_project"}  # Missing 'cookiecutter' key

    # Create a test file
    replay_file = get_file_name(replay_dir, template_name)
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test and Assert
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)


# LLM-generated content at query #174
#--------------------------

```python
def test_dump():
    # Setup
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    expected_file_path = os.path.join(replay_dir, f"{template_name}.json")

    # Exercise
    dump(replay_dir, template_name, context)

    # Verify
    assert os.path.exists(expected_file_path)
    with open(expected_file_path, encoding="utf-8") as infile:
        loaded_context = json.load(infile)
    assert loaded_context == context

    # Cleanup
    if os.path.exists(expected_file_path):
        os.remove(expected_file_path)
    if os.path.exists(replay_dir):
        os.rmdir(replay_dir)


# LLM-generated content at query #175
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test_project"}}

    # Write test file
    replay_file = get_file_name(replay_dir, template_name)
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test
    loaded_context = load(replay_dir, template_name)

    # Assert
    assert loaded_context == context


# LLM-generated content at query #176
#--------------------------

```python
def test_load(tmp_path):
    # Setup
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "test_project"}}

    # Create a test file
    replay_file = get_file_name(replay_dir, template_name)
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test
    loaded_context = load(replay_dir, template_name)

    # Assert
    assert loaded_context == context


# LLM-generated content at query #177
#--------------------------

```python
def test_dump():
    # Setup
    replay_dir = 'test_replay_dir'
    template_name = 'test_template'
    context = {'cookiecutter': {'key': 'value'}}
    expected_file = os.path.join(replay_dir, f'{template_name}.json')

    # Ensure the directory is clean before the test
    if os.path.exists(replay_dir):
        import shutil
        shutil.rmtree(replay_dir)

    # Execute
    dump(replay_dir, template_name, context)

    # Verify
    assert os.path.exists(expected_file)
    with open(expected_file, encoding="utf-8") as infile:
        loaded_context = json.load(infile)
    assert loaded_context == context

    # Cleanup
    if os.path.exists(replay_dir):
        import shutil
        shutil.rmtree(replay_dir)


# LLM-generated content at query #178
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


# LLM-generated content at query #179
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


# LLM-generated content at query #180
#--------------------------

```python
def test_dump(tmp_path):
    # Test basic functionality
    context = {'cookiecutter': {'project_name': 'test'}}
    template_name = 'test_template'
    replay_dir = tmp_path / 'replay'

    dump(replay_dir, template_name, context)

    # Verify file was created
    replay_file = get_file_name(replay_dir, template_name)
    assert os.path.exists(replay_file)

    # Verify content
    with open(replay_file, encoding="utf-8") as f:
        loaded_context = json.load(f)
    assert loaded_context == context

    # Test with .json extension
    template_name_json = 'test_template.json'
    dump(replay_dir, template_name_json, context)

    replay_file_json = get_file_name(replay_dir, template_name_json)
    assert os.path.exists(replay_file_json)

    # Test ValueError when cookiecutter key is missing
    invalid_context = {'project_name': 'test'}
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        dump(replay_dir, template_name, invalid_context)


# LLM-generated content at query #181
#--------------------------

```python
def test_dump(tmp_path):
    # Setup
    replay_dir = tmp_path / "replay"
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

def test_dump_missing_cookiecutter_key(tmp_path):
    # Setup
    replay_dir = tmp_path / "replay"
    template_name = "test_template"
    context = {"key": "value"}

    # Execute & Verify
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)

def test_dump_with_json_suffix(tmp_path):
    # Setup
    replay_dir = tmp_path / "replay"
    template_name = "test_template.json"
    context = {"cookiecutter": {"key": "value"}}

    # Execute
    dump(replay_dir, template_name, context)

    # Verify
    replay_file = get_file_name(replay_dir, template_name)
    assert os.path.exists(replay_file)

    with open(replay_file, encoding="utf-8") as infile:
        loaded_context = json.load(infile)

    assert loaded_context == context


# LLM-generated content at query #182
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


# LLM-generated content at query #183
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

    # Test
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


# LLM-generated content at query #184
#--------------------------

```python
def test_load():
    # Setup
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    file_name = get_file_name(replay_dir, template_name)
    os.makedirs(replay_dir, exist_ok=True)
    with open(file_name, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Test
    loaded_context = load(replay_dir, template_name)

    # Assert
    assert loaded_context == context

    # Test with .json suffix
    template_name_with_suffix = "test_template.json"
    file_name_with_suffix = get_file_name(replay_dir, template_name_with_suffix)
    with open(file_name_with_suffix, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    loaded_context_with_suffix = load(replay_dir, template_name_with_suffix)
    assert loaded_context_with_suffix == context

    # Test with missing cookiecutter key
    invalid_context = {"key": "value"}
    invalid_file_name = get_file_name(replay_dir, "invalid_template")
    with open(invalid_file_name, 'w', encoding="utf-8") as outfile:
        json.dump(invalid_context, outfile, indent=2)

    try:
        load(replay_dir, "invalid_template")
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "Context is required to contain a cookiecutter key"

    # Cleanup
    import shutil
    shutil.rmtree(replay_dir)


# LLM-generated content at query #185
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
    with open(expected_file, encoding="utf-8") as infile:
        loaded_context = json.load(infile)
    assert loaded_context == context

    # Cleanup
    os.remove(expected_file)
    os.rmdir(replay_dir)


