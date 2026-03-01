####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_get_file_name():
    # Test basic functionality with string path
    result = get_file_name("/tmp/replay", "my_template")
    assert result == "/tmp/replay/my_template.json"
    
    # Test with Path object
    from pathlib import Path
    result = get_file_name(Path("/tmp/replay"), "my_template")
    assert result == os.path.join("/tmp/replay", "my_template.json")
    
    # Test when template_name already has .json extension
    result = get_file_name("/tmp/replay", "my_template.json")
    assert result == "/tmp/replay/my_template.json"
    
    # Test with different extensions (should still add .json)
    result = get_file_name("/tmp/replay", "my_template.txt")
    assert result == "/tmp/replay/my_template.txt.json"
    
    # Test with empty template name
    result = get_file_name("/tmp/replay", "")
    assert result == "/tmp/replay/.json"
    
    # Test with nested paths in template_name
    result = get_file_name("/tmp/replay", "nested/template")
    assert result == "/tmp/replay/nested/template.json"
    
    # Test with Windows-style paths
    result = get_file_name("C:\\Users\\test", "my_template")
    assert result == os.path.join("C:\\Users\\test", "my_template.json")


# LLM-generated content at query #2
#--------------------------

```python
def test_get_file_name():
    # Test with template name without .json extension
    result = get_file_name("/tmp/replay", "my_template")
    assert result == "/tmp/replay/my_template.json"
    
    # Test with template name with .json extension
    result = get_file_name("/tmp/replay", "my_template.json")
    assert result == "/tmp/replay/my_template.json"
    
    # Test with Path object as replay_dir
    from pathlib import Path
    result = get_file_name(Path("/tmp/replay"), "my_template")
    assert result == str(Path("/tmp/replay/my_template.json"))
    
    # Test with Windows-style paths
    result = get_file_name("C:\\replay", "my_template")
    assert result == "C:\\replay\\my_template.json"
    
    # Test with relative path
    result = get_file_name("./replay", "my_template")
    assert result == "./replay/my_template.json"
    
    # Test with empty template name
    result = get_file_name("/tmp/replay", "")
    assert result == "/tmp/replay/.json"
    
    # Test with template name containing dots
    result = get_file_name("/tmp/replay", "my.template.name")
    assert result == "/tmp/replay/my.template.name.json"


# LLM-generated content at query #3
#--------------------------

```python
def test_load():
    import tempfile
    import json
    from pathlib import Path

    # Test normal case with valid context containing cookiecutter key
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "test_template"
        
        # Create test data
        test_context = {"cookiecutter": {"project_name": "Test Project", "version": "1.0"}}
        replay_file = replay_dir / f"{template_name}.json"
        
        # Write test data to file
        replay_file.write_text(json.dumps(test_context), encoding="utf-8")
        
        # Test load function
        result = load(replay_dir, template_name)
        assert result == test_context
        assert "cookiecutter" in result

    # Test with template_name already having .json extension
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "test_template.json"
        
        # Create test data
        test_context = {"cookiecutter": {"key": "value"}}
        replay_file = replay_dir / template_name
        
        # Write test data to file
        replay_file.write_text(json.dumps(test_context), encoding="utf-8")
        
        # Test load function
        result = load(replay_dir, template_name)
        assert result == test_context

    # Test with string path instead of Path object
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = "another_template"
        
        # Create test data
        test_context = {"cookiecutter": {"name": "Test"}}
        replay_file = Path(tmpdir) / f"{template_name}.json"
        
        # Write test data to file
        replay_file.write_text(json.dumps(test_context), encoding="utf-8")
        
        # Test load function with string path
        result = load(tmpdir, template_name)
        assert result == test_context

    # Test error case: missing cookiecutter key
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "invalid_template"
        
        # Create invalid test data (missing cookiecutter key)
        invalid_context = {"project_name": "Test Project"}
        replay_file = replay_dir / f"{template_name}.json"
        
        # Write invalid data to file
        replay_file.write_text(json.dumps(invalid_context), encoding="utf-8")
        
        # Test that ValueError is raised
        try:
            load(replay_dir, template_name)
            assert False, "Expected ValueError for missing cookiecutter key"
        except ValueError as e:
            assert "Context is required to contain a cookiecutter key" in str(e)

    # Test error case: file not found
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "non_existent_template"
        
        # Test that FileNotFoundError is raised
        try:
            load(replay_dir, template_name)
            assert False, "Expected FileNotFoundError"
        except FileNotFoundError:
            pass  # Expected behavior


# LLM-generated content at query #4
#--------------------------

```python
def test_dump():
    import tempfile
    import json
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = tmpdir
        template_name = "test_template"
        
        context = {"cookiecutter": {"project_name": "Test Project", "author": "Test Author"}}
        
        dump(replay_dir, template_name, context)
        
        expected_file = os.path.join(replay_dir, f"{template_name}.json")
        assert os.path.exists(expected_file)
        
        with open(expected_file, "r", encoding="utf-8") as f:
            loaded_data = json.load(f)
        
        assert loaded_data == context
        
        context_with_json = {"cookiecutter": {"key": "value"}}
        template_with_json = "template.json"
        
        dump(replay_dir, template_with_json, context_with_json)
        
        expected_file_json = os.path.join(replay_dir, template_with_json)
        assert os.path.exists(expected_file_json)
        
        with open(expected_file_json, "r", encoding="utf-8") as f:
            loaded_data_json = json.load(f)
        
        assert loaded_data_json == context_with_json


def test_dump_missing_cookiecutter():
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = tmpdir
        template_name = "test_template"
        
        context = {"project_name": "Test Project", "author": "Test Author"}
        
        try:
            dump(replay_dir, template_name, context)
            assert False, "Expected ValueError"
        except ValueError as e:
            assert str(e) == "Context is required to contain a cookiecutter key"


def test_dump_creates_directory():
    import tempfile
    import json
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = os.path.join(tmpdir, "nested", "directory")
        template_name = "test_template"
        
        context = {"cookiecutter": {"key": "value"}}
        
        assert not os.path.exists(replay_dir)
        
        dump(replay_dir, template_name, context)
        
        assert os.path.exists(replay_dir)
        
        expected_file = os.path.join(replay_dir, f"{template_name}.json")
        assert os.path.exists(expected_file)
        
        with open(expected_file, "r", encoding="utf-8") as f:
            loaded_data = json.load(f)
        
        assert loaded_data == context


# LLM-generated content at query #5
#--------------------------

```python
def test_dump():
    import tempfile
    import json
    import os
    
    # Test 1: Normal dump with valid context
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = "my_template"
        context = {"cookiecutter": {"project_name": "Test Project", "version": "1.0"}}
        
        dump(tmpdir, template_name, context)
        
        expected_file = os.path.join(tmpdir, f"{template_name}.json")
        assert os.path.exists(expected_file)
        
        with open(expected_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        assert loaded_data == context
    
    # Test 2: Dump with template_name ending in .json
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = "my_template.json"
        context = {"cookiecutter": {"key": "value"}}
        
        dump(tmpdir, template_name, context)
        
        expected_file = os.path.join(tmpdir, template_name)
        assert os.path.exists(expected_file)
        
        with open(expected_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        assert loaded_data == context
    
    # Test 3: Dump creates directory if it doesn't exist
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = os.path.join(tmpdir, "subdir", "nested")
        template_name = "test"
        context = {"cookiecutter": {"data": "test"}}
        
        assert not os.path.exists(replay_dir)
        dump(replay_dir, template_name, context)
        
        expected_file = os.path.join(replay_dir, f"{template_name}.json")
        assert os.path.exists(expected_file)
    
    # Test 4: Dump raises ValueError when context lacks 'cookiecutter' key
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = "invalid"
        context = {"not_cookiecutter": {"key": "value"}}
        
        try:
            dump(tmpdir, template_name, context)
            assert False, "Expected ValueError"
        except ValueError as e:
            assert "Context is required to contain a cookiecutter key" in str(e)
    
    # Test 5: Dump with complex nested context
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = "complex"
        context = {
            "cookiecutter": {
                "project": {
                    "name": "Test",
                    "settings": {"debug": True, "port": 8080}
                },
                "author": "Test Author",
                "dependencies": ["dep1", "dep2"]
            }
        }
        
        dump(tmpdir, template_name, context)
        
        expected_file = os.path.join(tmpdir, f"{template_name}.json")
        with open(expected_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        assert loaded_data == context


# LLM-generated content at query #6
#--------------------------

```python
def test_dump():
    import tempfile
    import json
    import os
    
    # Test 1: Normal dump with valid context
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = "my_template"
        context = {"cookiecutter": {"project_name": "Test Project", "version": "1.0"}}
        
        dump(tmpdir, template_name, context)
        
        expected_file = os.path.join(tmpdir, f"{template_name}.json")
        assert os.path.exists(expected_file)
        
        with open(expected_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        assert loaded_data == context
    
    # Test 2: Dump with template name ending in .json
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = "my_template.json"
        context = {"cookiecutter": {"key": "value"}}
        
        dump(tmpdir, template_name, context)
        
        expected_file = os.path.join(tmpdir, template_name)
        assert os.path.exists(expected_file)
        
        with open(expected_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        assert loaded_data == context
    
    # Test 3: Dump raises ValueError when cookiecutter key is missing
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = "test_template"
        context = {"not_cookiecutter": {"key": "value"}}
        
        try:
            dump(tmpdir, template_name, context)
            assert False, "Expected ValueError"
        except ValueError as e:
            assert "Context is required to contain a cookiecutter key" in str(e)
    
    # Test 4: Dump creates directory if it doesn't exist
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = os.path.join(tmpdir, "nonexistent", "subdir")
        template_name = "template"
        context = {"cookiecutter": {"test": "data"}}
        
        assert not os.path.exists(replay_dir)
        dump(replay_dir, template_name, context)
        assert os.path.exists(replay_dir)
        
        expected_file = os.path.join(replay_dir, f"{template_name}.json")
        assert os.path.exists(expected_file)
    
    # Test 5: Dump with nested context
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = "complex_template"
        context = {
            "cookiecutter": {
                "project": {
                    "name": "Test",
                    "author": "Developer",
                    "dependencies": ["dep1", "dep2"]
                },
                "choices": [1, 2, 3],
                "flag": True
            }
        }
        
        dump(tmpdir, template_name, context)
        
        expected_file = os.path.join(tmpdir, f"{template_name}.json")
        with open(expected_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        assert loaded_data == context


# LLM-generated content at query #7
#--------------------------

```python
def test_load():
    import tempfile
    import json
    from pathlib import Path

    # Test normal case with valid JSON containing cookiecutter key
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "test_template"
        
        # Create test data
        test_context = {"cookiecutter": {"project_name": "Test Project", "version": "1.0"}}
        
        # Write test file
        replay_file = replay_dir / f"{template_name}.json"
        with open(replay_file, "w", encoding="utf-8") as f:
            json.dump(test_context, f, indent=2)
        
        # Test load function
        result = load(replay_dir, template_name)
        assert result == test_context
        assert "cookiecutter" in result
        assert result["cookiecutter"]["project_name"] == "Test Project"

    # Test with template name that already has .json extension
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "test_template.json"
        
        test_context = {"cookiecutter": {"key": "value"}}
        
        replay_file = replay_dir / template_name
        with open(replay_file, "w", encoding="utf-8") as f:
            json.dump(test_context, f, indent=2)
        
        result = load(replay_dir, template_name)
        assert result == test_context

    # Test with string path instead of Path object
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = "another_template"
        
        test_context = {"cookiecutter": {"data": "test"}}
        
        replay_file = Path(tmpdir) / f"{template_name}.json"
        with open(replay_file, "w", encoding="utf-8") as f:
            json.dump(test_context, f, indent=2)
        
        result = load(tmpdir, template_name)
        assert result == test_context

    # Test error case: missing cookiecutter key
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "invalid_template"
        
        invalid_context = {"not_cookiecutter": {"key": "value"}}
        
        replay_file = replay_dir / f"{template_name}.json"
        with open(replay_file, "w", encoding="utf-8") as f:
            json.dump(invalid_context, f, indent=2)
        
        try:
            load(replay_dir, template_name)
            assert False, "Expected ValueError for missing cookiecutter key"
        except ValueError as e:
            assert "Context is required to contain a cookiecutter key" in str(e)

    # Test error case: file not found
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "non_existent"
        
        try:
            load(replay_dir, template_name)
            assert False, "Expected FileNotFoundError"
        except FileNotFoundError:
            pass

    # Test with nested cookiecutter data
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "nested_template"
        
        nested_context = {
            "cookiecutter": {
                "project": {
                    "name": "Test",
                    "author": "Developer"
                },
                "settings": {
                    "debug": True,
                    "port": 8000
                }
            }
        }
        
        replay_file = replay_dir / f"{template_name}.json"
        with open(replay_file, "w", encoding="utf-8") as f:
            json.dump(nested_context, f, indent=2)
        
        result = load(replay_dir, template_name)
        assert result == nested_context
        assert result["cookiecutter"]["project"]["name"] == "Test"
        assert result["cookiecutter"]["settings"]["debug"] is True


# LLM-generated content at query #8
#--------------------------

```python
def test_load():
    import tempfile
    import json
    from pathlib import Path

    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "test_template"
        
        # Test 1: Normal case with valid context containing 'cookiecutter' key
        test_context = {
            "cookiecutter": {
                "project_name": "Test Project",
                "author": "Test Author",
                "version": "1.0.0"
            }
        }
        
        # Create the replay file
        replay_file = replay_dir / f"{template_name}.json"
        with open(replay_file, 'w', encoding='utf-8') as f:
            json.dump(test_context, f, indent=2)
        
        # Load and verify
        result = load(replay_dir, template_name)
        assert result == test_context
        assert "cookiecutter" in result
        
        # Test 2: Template name without .json extension
        template_name2 = "template2"
        replay_file2 = replay_dir / f"{template_name2}.json"
        with open(replay_file2, 'w', encoding='utf-8') as f:
            json.dump(test_context, f, indent=2)
        
        result2 = load(replay_dir, template_name2)
        assert result2 == test_context
        
        # Test 3: Template name with .json extension
        template_name3 = "template3.json"
        replay_file3 = replay_dir / "template3.json"
        with open(replay_file3, 'w', encoding='utf-8') as f:
            json.dump(test_context, f, indent=2)
        
        result3 = load(replay_dir, template_name3)
        assert result3 == test_context
        
        # Test 4: Missing 'cookiecutter' key should raise ValueError
        invalid_context = {"project_name": "Test", "author": "Author"}
        replay_file4 = replay_dir / "invalid.json"
        with open(replay_file4, 'w', encoding='utf-8') as f:
            json.dump(invalid_context, f, indent=2)
        
        try:
            load(replay_dir, "invalid")
            assert False, "Expected ValueError for missing 'cookiecutter' key"
        except ValueError as e:
            assert "Context is required to contain a cookiecutter key" in str(e)
        
        # Test 5: Non-existent file should raise FileNotFoundError
        try:
            load(replay_dir, "nonexistent")
            assert False, "Expected FileNotFoundError"
        except FileNotFoundError:
            pass
        
        # Test 6: Invalid JSON should raise JSONDecodeError
        replay_file6 = replay_dir / "bad_json.json"
        with open(replay_file6, 'w', encoding='utf-8') as f:
            f.write("{invalid json")
        
        try:
            load(replay_dir, "bad_json")
            assert False, "Expected JSONDecodeError"
        except json.JSONDecodeError:
            pass


# LLM-generated content at query #9
#--------------------------

```python
def test_dump(tmp_path):
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {"cookiecutter": {"project_name": "Test Project", "version": "1.0"}}

    dump(replay_dir, template_name, context)

    expected_file = replay_dir / f"{template_name}.json"
    assert expected_file.exists()

    with open(expected_file, encoding="utf-8") as f:
        loaded_data = json.load(f)
    assert loaded_data == context


def test_dump_with_json_extension(tmp_path):
    replay_dir = tmp_path / "replay"
    template_name = "my_template.json"
    context = {"cookiecutter": {"key": "value"}}

    dump(replay_dir, template_name, context)

    expected_file = replay_dir / template_name
    assert expected_file.exists()

    with open(expected_file, encoding="utf-8") as f:
        loaded_data = json.load(f)
    assert loaded_data == context


def test_dump_missing_cookiecutter_key(tmp_path):
    replay_dir = tmp_path / "replay"
    template_name = "template"
    context = {"not_cookiecutter": {"key": "value"}}

    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


def test_dump_creates_directory(tmp_path):
    replay_dir = tmp_path / "new" / "nested" / "replay"
    template_name = "template"
    context = {"cookiecutter": {"key": "value"}}

    assert not replay_dir.exists()
    dump(replay_dir, template_name, context)
    assert replay_dir.exists()


def test_dump_with_complex_context(tmp_path):
    replay_dir = tmp_path / "replay"
    template_name = "complex_template"
    context = {
        "cookiecutter": {
            "project_name": "Test",
            "list_data": [1, 2, 3],
            "nested": {"inner": "value"},
            "boolean": True,
            "null": None,
        }
    }

    dump(replay_dir, template_name, context)

    expected_file = replay_dir / f"{template_name}.json"
    with open(expected_file, encoding="utf-8") as f:
        loaded_data = json.load(f)
    assert loaded_data == context


# LLM-generated content at query #10
#--------------------------

```python
def test_dump():
    import tempfile
    import json
    import os
    from pathlib import Path
    
    # Test 1: Normal dump with valid context
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "my_template"
        context = {"cookiecutter": {"project_name": "Test Project", "version": "1.0"}}
        
        dump(replay_dir, template_name, context)
        
        expected_file = replay_dir / f"{template_name}.json"
        assert expected_file.exists()
        
        with open(expected_file, "r", encoding="utf-8") as f:
            loaded_context = json.load(f)
        
        assert loaded_context == context
    
    # Test 2: Dump with existing .json extension in template_name
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "my_template.json"
        context = {"cookiecutter": {"key": "value"}}
        
        dump(replay_dir, template_name, context)
        
        expected_file = replay_dir / template_name
        assert expected_file.exists()
        
        with open(expected_file, "r", encoding="utf-8") as f:
            loaded_context = json.load(f)
        
        assert loaded_context == context
    
    # Test 3: Dump creates directory if it doesn't exist
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir) / "subdir" / "nested"
        template_name = "test"
        context = {"cookiecutter": {"data": "test"}}
        
        assert not replay_dir.exists()
        dump(replay_dir, template_name, context)
        assert replay_dir.exists()
    
    # Test 4: Dump raises ValueError when context lacks 'cookiecutter' key
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "test"
        context = {"not_cookiecutter": {"key": "value"}}
        
        try:
            dump(replay_dir, template_name, context)
            assert False, "Expected ValueError"
        except ValueError as e:
            assert "Context is required to contain a cookiecutter key" in str(e)
    
    # Test 5: Dump with string path instead of Path object
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = tmpdir
        template_name = "string_test"
        context = {"cookiecutter": {"test": "data"}}
        
        dump(replay_dir, template_name, context)
        
        expected_file = Path(replay_dir) / f"{template_name}.json"
        assert expected_file.exists()
        
        with open(expected_file, "r", encoding="utf-8") as f:
            loaded_context = json.load(f)
        
        assert loaded_context == context
    
    # Test 6: Dump with complex nested context
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "complex"
        context = {
            "cookiecutter": {
                "project": {
                    "name": "Test",
                    "settings": {"debug": True, "port": 8080}
                },
                "author": "Test Author",
                "tags": ["python", "testing"]
            }
        }
        
        dump(replay_dir, template_name, context)
        
        expected_file = replay_dir / f"{template_name}.json"
        assert expected_file.exists()
        
        with open(expected_file, "r", encoding="utf-8") as f:
            loaded_context = json.load(f)
        
        assert loaded_context == context


# LLM-generated content at query #11
#--------------------------

```python
def test_load():
    import tempfile
    import json
    from pathlib import Path

    # Test normal case with valid cookiecutter context
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "my_template"
        
        # Create test data
        test_context = {
            "cookiecutter": {
                "project_name": "Test Project",
                "version": "1.0.0",
                "author": "Test Author"
            }
        }
        
        # Write test file
        replay_file = replay_dir / f"{template_name}.json"
        with open(replay_file, "w", encoding="utf-8") as f:
            json.dump(test_context, f, indent=2)
        
        # Test load function
        result = load(replay_dir, template_name)
        assert result == test_context
        assert "cookiecutter" in result

    # Test with template name that already has .json extension
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "my_template.json"
        
        test_context = {
            "cookiecutter": {
                "key": "value"
            }
        }
        
        replay_file = replay_dir / template_name
        with open(replay_file, "w", encoding="utf-8") as f:
            json.dump(test_context, f, indent=2)
        
        result = load(replay_dir, template_name)
        assert result == test_context

    # Test with string path instead of Path object
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = "another_template"
        
        test_context = {
            "cookiecutter": {
                "name": "Test"
            }
        }
        
        replay_file = Path(tmpdir) / f"{template_name}.json"
        with open(replay_file, "w", encoding="utf-8") as f:
            json.dump(test_context, f, indent=2)
        
        result = load(tmpdir, template_name)
        assert result == test_context

    # Test error case: missing cookiecutter key
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "bad_template"
        
        bad_context = {
            "not_cookiecutter": {
                "key": "value"
            }
        }
        
        replay_file = replay_dir / f"{template_name}.json"
        with open(replay_file, "w", encoding="utf-8") as f:
            json.dump(bad_context, f, indent=2)
        
        try:
            load(replay_dir, template_name)
            assert False, "Expected ValueError for missing cookiecutter key"
        except ValueError as e:
            assert "Context is required to contain a cookiecutter key" in str(e)

    # Test error case: file not found
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "non_existent"
        
        try:
            load(replay_dir, template_name)
            assert False, "Expected FileNotFoundError"
        except FileNotFoundError:
            pass


# LLM-generated content at query #12
#--------------------------

```python
def test_load():
    import tempfile
    import json
    from pathlib import Path

    # Test normal case with valid context containing cookiecutter key
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "test_template"
        
        # Create valid replay file
        replay_file = replay_dir / f"{template_name}.json"
        valid_context = {"cookiecutter": {"project_name": "Test Project", "version": "1.0"}}
        
        with open(replay_file, "w", encoding="utf-8") as f:
            json.dump(valid_context, f)
        
        # Test loading valid context
        result = load(replay_dir, template_name)
        assert result == valid_context
        assert "cookiecutter" in result
        assert result["cookiecutter"]["project_name"] == "Test Project"

    # Test with template name that already has .json extension
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "test_template.json"
        
        replay_file = replay_dir / template_name
        valid_context = {"cookiecutter": {"key": "value"}}
        
        with open(replay_file, "w", encoding="utf-8") as f:
            json.dump(valid_context, f)
        
        result = load(replay_dir, template_name)
        assert result == valid_context

    # Test with template name without .json extension
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "another_template"
        
        replay_file = replay_dir / f"{template_name}.json"
        valid_context = {"cookiecutter": {"data": "test"}}
        
        with open(replay_file, "w", encoding="utf-8") as f:
            json.dump(valid_context, f)
        
        result = load(replay_dir, template_name)
        assert result == valid_context

    # Test error case: missing cookiecutter key
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "invalid_template"
        
        replay_file = replay_dir / f"{template_name}.json"
        invalid_context = {"not_cookiecutter": {"key": "value"}}
        
        with open(replay_file, "w", encoding="utf-8") as f:
            json.dump(invalid_context, f)
        
        try:
            load(replay_dir, template_name)
            assert False, "Expected ValueError for missing cookiecutter key"
        except ValueError as e:
            assert "Context is required to contain a cookiecutter key" in str(e)

    # Test error case: file not found
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "non_existent"
        
        try:
            load(replay_dir, template_name)
            assert False, "Expected FileNotFoundError"
        except FileNotFoundError:
            pass

    # Test with string path instead of Path object
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = "string_path_test"
        
        replay_file = Path(tmpdir) / f"{template_name}.json"
        valid_context = {"cookiecutter": {"test": "data"}}
        
        with open(replay_file, "w", encoding="utf-8") as f:
            json.dump(valid_context, f)
        
        result = load(tmpdir, template_name)
        assert result == valid_context


# LLM-generated content at query #13
#--------------------------

```python
def test_dump():
    import tempfile
    import json
    import os
    
    # Test 1: Normal dump with valid context
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = "my_template"
        context = {"cookiecutter": {"project_name": "Test Project", "version": "1.0"}}
        
        dump(tmpdir, template_name, context)
        
        expected_file = os.path.join(tmpdir, f"{template_name}.json")
        assert os.path.exists(expected_file)
        
        with open(expected_file, "r", encoding="utf-8") as f:
            loaded_context = json.load(f)
        
        assert loaded_context == context
    
    # Test 2: Dump with template_name ending in .json
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = "my_template.json"
        context = {"cookiecutter": {"key": "value"}}
        
        dump(tmpdir, template_name, context)
        
        expected_file = os.path.join(tmpdir, template_name)
        assert os.path.exists(expected_file)
        
        with open(expected_file, "r", encoding="utf-8") as f:
            loaded_context = json.load(f)
        
        assert loaded_context == context
    
    # Test 3: Dump raises ValueError when context lacks 'cookiecutter' key
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = "test_template"
        context = {"not_cookiecutter": {"key": "value"}}
        
        try:
            dump(tmpdir, template_name, context)
            assert False, "Expected ValueError"
        except ValueError as e:
            assert str(e) == "Context is required to contain a cookiecutter key"
    
    # Test 4: Dump creates directory if it doesn't exist
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = os.path.join(tmpdir, "nonexistent", "subdir")
        template_name = "template"
        context = {"cookiecutter": {"test": "data"}}
        
        dump(replay_dir, template_name, context)
        
        expected_file = os.path.join(replay_dir, f"{template_name}.json")
        assert os.path.exists(expected_file)
    
    # Test 5: Dump with nested context structure
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = "complex_template"
        context = {
            "cookiecutter": {
                "project": {
                    "name": "Test",
                    "author": "Developer",
                    "dependencies": ["dep1", "dep2"]
                },
                "settings": {"debug": True, "port": 8080}
            }
        }
        
        dump(tmpdir, template_name, context)
        
        expected_file = os.path.join(tmpdir, f"{template_name}.json")
        assert os.path.exists(expected_file)
        
        with open(expected_file, "r", encoding="utf-8") as f:
            loaded_context = json.load(f)
        
        assert loaded_context == context


# LLM-generated content at query #14
#--------------------------

```python
def test_load():
    import tempfile
    import json
    from pathlib import Path

    # Test normal case with valid JSON containing cookiecutter key
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "test_template"
        
        # Create test data
        test_context = {"cookiecutter": {"project_name": "Test Project", "version": "1.0"}}
        replay_file = replay_dir / f"{template_name}.json"
        
        # Write test file
        replay_file.write_text(json.dumps(test_context), encoding="utf-8")
        
        # Test load function
        result = load(replay_dir, template_name)
        assert result == test_context
        assert "cookiecutter" in result

    # Test with template name already having .json extension
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "test_template.json"
        
        test_context = {"cookiecutter": {"key": "value"}}
        replay_file = replay_dir / template_name
        
        replay_file.write_text(json.dumps(test_context), encoding="utf-8")
        
        result = load(replay_dir, template_name)
        assert result == test_context

    # Test with string path instead of Path object
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = "another_template"
        
        test_context = {"cookiecutter": {"name": "Test"}}
        replay_file = Path(tmpdir) / f"{template_name}.json"
        
        replay_file.write_text(json.dumps(test_context), encoding="utf-8")
        
        result = load(tmpdir, template_name)
        assert result == test_context

    # Test error case: missing cookiecutter key
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "bad_template"
        
        bad_context = {"not_cookiecutter": {"key": "value"}}
        replay_file = replay_dir / f"{template_name}.json"
        
        replay_file.write_text(json.dumps(bad_context), encoding="utf-8")
        
        try:
            load(replay_dir, template_name)
            assert False, "Expected ValueError for missing cookiecutter key"
        except ValueError as e:
            assert "Context is required to contain a cookiecutter key" in str(e)

    # Test error case: file not found
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "nonexistent"
        
        try:
            load(replay_dir, template_name)
            assert False, "Expected FileNotFoundError"
        except FileNotFoundError:
            pass

    # Test error case: invalid JSON
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "invalid_json"
        
        replay_file = replay_dir / f"{template_name}.json"
        replay_file.write_text("{invalid json", encoding="utf-8")
        
        try:
            load(replay_dir, template_name)
            assert False, "Expected JSON decode error"
        except json.JSONDecodeError:
            pass


# LLM-generated content at query #15
#--------------------------

```python
def test_dump(tmp_path):
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {"cookiecutter": {"project_name": "Test Project", "version": "1.0"}}

    dump(replay_dir, template_name, context)

    expected_file = replay_dir / f"{template_name}.json"
    assert expected_file.exists()

    with open(expected_file, encoding="utf-8") as f:
        loaded_data = json.load(f)
    assert loaded_data == context


def test_dump_with_json_extension(tmp_path):
    replay_dir = tmp_path / "replay"
    template_name = "my_template.json"
    context = {"cookiecutter": {"key": "value"}}

    dump(replay_dir, template_name, context)

    expected_file = replay_dir / template_name
    assert expected_file.exists()

    with open(expected_file, encoding="utf-8") as f:
        loaded_data = json.load(f)
    assert loaded_data == context


def test_dump_missing_cookiecutter_key(tmp_path):
    replay_dir = tmp_path / "replay"
    template_name = "template"
    context = {"not_cookiecutter": {"key": "value"}}

    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


def test_dump_creates_directory(tmp_path):
    replay_dir = tmp_path / "new" / "nested" / "replay"
    template_name = "template"
    context = {"cookiecutter": {"test": "data"}}

    assert not replay_dir.exists()
    dump(replay_dir, template_name, context)
    assert replay_dir.exists()


def test_dump_with_complex_context(tmp_path):
    replay_dir = tmp_path / "replay"
    template_name = "complex_template"
    context = {
        "cookiecutter": {
            "name": "Test",
            "list": [1, 2, 3],
            "nested": {"key": "value"},
            "boolean": True,
            "null": None,
            "number": 42,
        }
    }

    dump(replay_dir, template_name, context)

    expected_file = replay_dir / f"{template_name}.json"
    with open(expected_file, encoding="utf-8") as f:
        loaded_data = json.load(f)
    assert loaded_data == context


# LLM-generated content at query #16
#--------------------------

```python
def test_load():
    import tempfile
    import json
    from pathlib import Path

    # Test 1: Normal case with valid context containing 'cookiecutter' key
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "test_template"
        
        # Create test data
        test_context = {
            "cookiecutter": {
                "project_name": "Test Project",
                "author": "Test Author"
            },
            "other_data": "test"
        }
        
        # Write test file
        replay_file = replay_dir / f"{template_name}.json"
        with open(replay_file, "w", encoding="utf-8") as f:
            json.dump(test_context, f)
        
        # Load and verify
        result = load(replay_dir, template_name)
        assert result == test_context
        assert "cookiecutter" in result

    # Test 2: File without .json extension in template_name
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "test_template"
        
        test_context = {
            "cookiecutter": {
                "key": "value"
            }
        }
        
        replay_file = replay_dir / f"{template_name}.json"
        with open(replay_file, "w", encoding="utf-8") as f:
            json.dump(test_context, f)
        
        result = load(replay_dir, template_name)
        assert result == test_context

    # Test 3: File with .json extension in template_name
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "test_template.json"
        
        test_context = {
            "cookiecutter": {
                "key": "value"
            }
        }
        
        replay_file = replay_dir / template_name
        with open(replay_file, "w", encoding="utf-8") as f:
            json.dump(test_context, f)
        
        result = load(replay_dir, template_name)
        assert result == test_context

    # Test 4: Missing 'cookiecutter' key should raise ValueError
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "invalid_template"
        
        invalid_context = {
            "not_cookiecutter": {
                "key": "value"
            }
        }
        
        replay_file = replay_dir / f"{template_name}.json"
        with open(replay_file, "w", encoding="utf-8") as f:
            json.dump(invalid_context, f)
        
        try:
            load(replay_dir, template_name)
            assert False, "Expected ValueError for missing cookiecutter key"
        except ValueError as e:
            assert "Context is required to contain a cookiecutter key" in str(e)

    # Test 5: Non-existent file should raise FileNotFoundError
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "non_existent"
        
        try:
            load(replay_dir, template_name)
            assert False, "Expected FileNotFoundError"
        except FileNotFoundError:
            pass

    # Test 6: Invalid JSON file should raise JSONDecodeError
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "invalid_json"
        
        replay_file = replay_dir / f"{template_name}.json"
        with open(replay_file, "w", encoding="utf-8") as f:
            f.write("invalid json content")
        
        try:
            load(replay_dir, template_name)
            assert False, "Expected JSONDecodeError"
        except json.JSONDecodeError:
            pass

    # Test 7: String path instead of Path object
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = tmpdir
        template_name = "string_path_test"
        
        test_context = {
            "cookiecutter": {
                "test": "value"
            }
        }
        
        replay_file = Path(replay_dir) / f"{template_name}.json"
        with open(replay_file, "w", encoding="utf-8") as f:
            json.dump(test_context, f)
        
        result = load(replay_dir, template_name)
        assert result == test_context


# LLM-generated content at query #17
#--------------------------

```python
def test_dump():
    import tempfile
    import json
    import os
    
    # Test 1: Normal dump with valid context
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = "my_template"
        context = {"cookiecutter": {"project_name": "Test Project", "version": "1.0"}}
        
        dump(tmpdir, template_name, context)
        
        expected_file = os.path.join(tmpdir, f"{template_name}.json")
        assert os.path.exists(expected_file)
        
        with open(expected_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        assert loaded_data == context
    
    # Test 2: Dump with template_name ending in .json
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = "my_template.json"
        context = {"cookiecutter": {"key": "value"}}
        
        dump(tmpdir, template_name, context)
        
        expected_file = os.path.join(tmpdir, template_name)
        assert os.path.exists(expected_file)
        
        with open(expected_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        assert loaded_data == context
    
    # Test 3: Dump creates directory if it doesn't exist
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = os.path.join(tmpdir, "subdir", "nested")
        template_name = "test_template"
        context = {"cookiecutter": {"test": "data"}}
        
        assert not os.path.exists(replay_dir)
        dump(replay_dir, template_name, context)
        assert os.path.exists(replay_dir)
        
        expected_file = os.path.join(replay_dir, f"{template_name}.json")
        assert os.path.exists(expected_file)
    
    # Test 4: Dump raises ValueError when context doesn't contain 'cookiecutter' key
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = "invalid_template"
        context = {"not_cookiecutter": {"key": "value"}}
        
        try:
            dump(tmpdir, template_name, context)
            assert False, "Expected ValueError"
        except ValueError as e:
            assert "Context is required to contain a cookiecutter key" in str(e)
    
    # Test 5: Dump with complex nested context
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = "complex_template"
        context = {
            "cookiecutter": {
                "project": {
                    "name": "Test",
                    "authors": ["Alice", "Bob"],
                    "config": {"debug": True, "port": 8080}
                },
                "choices": [1, 2, 3],
                "description": "A test project"
            }
        }
        
        dump(tmpdir, template_name, context)
        
        expected_file = os.path.join(tmpdir, f"{template_name}.json")
        with open(expected_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        assert loaded_data == context
        assert loaded_data["cookiecutter"]["project"]["authors"] == ["Alice", "Bob"]
        assert loaded_data["cookiecutter"]["project"]["config"]["debug"] is True
    
    # Test 6: Dump with Path object as replay_dir
    with tempfile.TemporaryDirectory() as tmpdir:
        from pathlib import Path
        replay_dir = Path(tmpdir)
        template_name = "path_template"
        context = {"cookiecutter": {"test": "path_object"}}
        
        dump(replay_dir, template_name, context)
        
        expected_file = os.path.join(tmpdir, f"{template_name}.json")
        assert os.path.exists(expected_file)
        
        with open(expected_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        assert loaded_data == context


# LLM-generated content at query #18
#--------------------------

```python
def test_dump():
    import tempfile
    import json
    import os
    
    # Test 1: Normal dump with valid context
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = "my_template"
        context = {"cookiecutter": {"project_name": "Test Project", "version": "1.0"}}
        
        dump(tmpdir, template_name, context)
        
        expected_file = os.path.join(tmpdir, f"{template_name}.json")
        assert os.path.exists(expected_file)
        
        with open(expected_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        assert loaded_data == context
    
    # Test 2: Dump with template name ending in .json
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = "my_template.json"
        context = {"cookiecutter": {"key": "value"}}
        
        dump(tmpdir, template_name, context)
        
        expected_file = os.path.join(tmpdir, template_name)
        assert os.path.exists(expected_file)
        
        with open(expected_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        assert loaded_data == context
    
    # Test 3: Dump raises ValueError when cookiecutter key is missing
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = "test_template"
        context = {"not_cookiecutter": {"key": "value"}}
        
        try:
            dump(tmpdir, template_name, context)
            assert False, "Expected ValueError to be raised"
        except ValueError as e:
            assert "Context is required to contain a cookiecutter key" in str(e)
    
    # Test 4: Dump creates directory if it doesn't exist
    with tempfile.TemporaryDirectory() as tmpdir:
        new_dir = os.path.join(tmpdir, "new_subdirectory")
        template_name = "template"
        context = {"cookiecutter": {"test": "data"}}
        
        assert not os.path.exists(new_dir)
        dump(new_dir, template_name, context)
        assert os.path.exists(new_dir)
        
        expected_file = os.path.join(new_dir, f"{template_name}.json")
        assert os.path.exists(expected_file)
    
    # Test 5: Dump with nested context
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = "complex_template"
        context = {
            "cookiecutter": {
                "project": {
                    "name": "Test",
                    "author": "Developer",
                    "dependencies": ["dep1", "dep2"]
                },
                "settings": {"debug": True, "port": 8000}
            }
        }
        
        dump(tmpdir, template_name, context)
        
        expected_file = os.path.join(tmpdir, f"{template_name}.json")
        with open(expected_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        assert loaded_data == context


# LLM-generated content at query #19
#--------------------------

```python
def test_dump():
    import tempfile
    import json
    import os
    
    # Test 1: Normal dump with valid context
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = "my_template"
        context = {"cookiecutter": {"project_name": "Test Project", "version": "1.0"}}
        
        dump(tmpdir, template_name, context)
        
        expected_file = os.path.join(tmpdir, f"{template_name}.json")
        assert os.path.exists(expected_file)
        
        with open(expected_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        assert loaded_data == context
    
    # Test 2: Dump with existing .json extension in template name
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = "my_template.json"
        context = {"cookiecutter": {"key": "value"}}
        
        dump(tmpdir, template_name, context)
        
        expected_file = os.path.join(tmpdir, template_name)
        assert os.path.exists(expected_file)
        
        with open(expected_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        assert loaded_data == context
    
    # Test 3: Dump raises ValueError when cookiecutter key is missing
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = "test_template"
        context = {"not_cookiecutter": {"key": "value"}}
        
        try:
            dump(tmpdir, template_name, context)
            assert False, "Expected ValueError to be raised"
        except ValueError as e:
            assert "Context is required to contain a cookiecutter key" in str(e)
    
    # Test 4: Dump creates directory if it doesn't exist
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = os.path.join(tmpdir, "nonexistent", "subdir")
        template_name = "template"
        context = {"cookiecutter": {"test": "data"}}
        
        assert not os.path.exists(replay_dir)
        dump(replay_dir, template_name, context)
        assert os.path.exists(replay_dir)
        
        expected_file = os.path.join(replay_dir, f"{template_name}.json")
        assert os.path.exists(expected_file)
    
    # Test 5: Dump with nested context structure
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = "complex_template"
        context = {
            "cookiecutter": {
                "project": {
                    "name": "Test",
                    "author": "Developer",
                    "dependencies": ["dep1", "dep2"]
                },
                "choices": {"license": "MIT", "python_version": "3.8"}
            }
        }
        
        dump(tmpdir, template_name, context)
        
        expected_file = os.path.join(tmpdir, f"{template_name}.json")
        with open(expected_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        assert loaded_data == context


# LLM-generated content at query #20
#--------------------------

```python
def test_dump(tmp_path):
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {"cookiecutter": {"project_name": "Test Project", "version": "1.0"}}

    dump(replay_dir, template_name, context)

    expected_file = replay_dir / f"{template_name}.json"
    assert expected_file.exists()

    with open(expected_file, encoding="utf-8") as f:
        loaded_data = json.load(f)
    assert loaded_data == context


def test_dump_with_json_extension(tmp_path):
    replay_dir = tmp_path / "replay"
    template_name = "my_template.json"
    context = {"cookiecutter": {"key": "value"}}

    dump(replay_dir, template_name, context)

    expected_file = replay_dir / template_name
    assert expected_file.exists()

    with open(expected_file, encoding="utf-8") as f:
        loaded_data = json.load(f)
    assert loaded_data == context


def test_dump_missing_cookiecutter_key(tmp_path):
    replay_dir = tmp_path / "replay"
    template_name = "template"
    context = {"not_cookiecutter": {"key": "value"}}

    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


def test_dump_creates_directory(tmp_path):
    replay_dir = tmp_path / "new" / "nested" / "replay"
    template_name = "template"
    context = {"cookiecutter": {"key": "value"}}

    assert not replay_dir.exists()
    dump(replay_dir, template_name, context)
    assert replay_dir.exists()


def test_dump_with_complex_context(tmp_path):
    replay_dir = tmp_path / "replay"
    template_name = "complex_template"
    context = {
        "cookiecutter": {
            "project_name": "Test",
            "choices": ["opt1", "opt2"],
            "nested": {"inner": "value"},
            "number": 42,
            "boolean": True,
            "null": None,
        }
    }

    dump(replay_dir, template_name, context)

    expected_file = replay_dir / f"{template_name}.json"
    with open(expected_file, encoding="utf-8") as f:
        loaded_data = json.load(f)
    assert loaded_data == context


# LLM-generated content at query #21
#--------------------------

```python
def test_load():
    import json
    import tempfile
    from pathlib import Path
    
    # Test 1: Normal load with valid context containing cookiecutter key
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "test_template"
        
        # Create valid context file
        context = {"cookiecutter": {"project_name": "Test Project", "version": "1.0"}}
        replay_file = replay_dir / f"{template_name}.json"
        
        with open(replay_file, "w", encoding="utf-8") as f:
            json.dump(context, f)
        
        # Load and verify
        loaded_context = load(replay_dir, template_name)
        assert loaded_context == context
        assert "cookiecutter" in loaded_context
        assert loaded_context["cookiecutter"]["project_name"] == "Test Project"
    
    # Test 2: Load with template name already having .json extension
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "test_template.json"
        
        context = {"cookiecutter": {"key": "value"}}
        replay_file = replay_dir / template_name
        
        with open(replay_file, "w", encoding="utf-8") as f:
            json.dump(context, f)
        
        loaded_context = load(replay_dir, template_name)
        assert loaded_context == context
    
    # Test 3: Load with string path instead of Path object
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = "template"
        
        context = {"cookiecutter": {"data": "test"}}
        replay_file = Path(tmpdir) / f"{template_name}.json"
        
        with open(replay_file, "w", encoding="utf-8") as f:
            json.dump(context, f)
        
        loaded_context = load(tmpdir, template_name)
        assert loaded_context == context
    
    # Test 4: Raise ValueError when cookiecutter key is missing
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "invalid_template"
        
        # Create invalid context without cookiecutter key
        invalid_context = {"project_name": "Test", "version": "1.0"}
        replay_file = replay_dir / f"{template_name}.json"
        
        with open(replay_file, "w", encoding="utf-8") as f:
            json.dump(invalid_context, f)
        
        try:
            load(replay_dir, template_name)
            assert False, "Expected ValueError"
        except ValueError as e:
            assert "Context is required to contain a cookiecutter key" in str(e)
    
    # Test 5: File not found (should raise FileNotFoundError)
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "non_existent"
        
        try:
            load(replay_dir, template_name)
            assert False, "Expected FileNotFoundError"
        except FileNotFoundError:
            pass  # Expected
    
    # Test 6: Load with nested cookiecutter data
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "nested_template"
        
        context = {
            "cookiecutter": {
                "project": {
                    "name": "Test",
                    "author": "Developer"
                },
                "settings": {
                    "debug": True,
                    "port": 8000
                }
            }
        }
        replay_file = replay_dir / f"{template_name}.json"
        
        with open(replay_file, "w", encoding="utf-8") as f:
            json.dump(context, f, indent=2)
        
        loaded_context = load(replay_dir, template_name)
        assert loaded_context == context
        assert loaded_context["cookiecutter"]["project"]["name"] == "Test"
        assert loaded_context["cookiecutter"]["settings"]["debug"] is True


# LLM-generated content at query #22
#--------------------------

```python
def test_load():
    import tempfile
    import json
    from pathlib import Path

    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "test_template"
        
        # Test 1: Normal case with valid context containing 'cookiecutter' key
        test_context = {
            "cookiecutter": {
                "project_name": "Test Project",
                "author": "Test Author",
                "version": "1.0.0"
            }
        }
        
        # Write test file
        replay_file = replay_dir / f"{template_name}.json"
        with open(replay_file, 'w', encoding='utf-8') as f:
            json.dump(test_context, f, indent=2)
        
        # Load and verify
        result = load(replay_dir, template_name)
        assert result == test_context
        assert "cookiecutter" in result
        
        # Test 2: Template name without .json extension
        template_name2 = "template2"
        test_context2 = {
            "cookiecutter": {
                "name": "Another Test"
            }
        }
        
        replay_file2 = replay_dir / f"{template_name2}.json"
        with open(replay_file2, 'w', encoding='utf-8') as f:
            json.dump(test_context2, f, indent=2)
        
        result2 = load(replay_dir, template_name2)
        assert result2 == test_context2
        
        # Test 3: Template name with .json extension
        template_name3 = "template3.json"
        test_context3 = {
            "cookiecutter": {
                "key": "value"
            }
        }
        
        replay_file3 = replay_dir / template_name3
        with open(replay_file3, 'w', encoding='utf-8') as f:
            json.dump(test_context3, f, indent=2)
        
        result3 = load(replay_dir, template_name3)
        assert result3 == test_context3
        
        # Test 4: Missing 'cookiecutter' key should raise ValueError
        invalid_context = {
            "not_cookiecutter": {
                "key": "value"
            }
        }
        
        template_name4 = "invalid_template"
        replay_file4 = replay_dir / f"{template_name4}.json"
        with open(replay_file4, 'w', encoding='utf-8') as f:
            json.dump(invalid_context, f, indent=2)
        
        try:
            load(replay_dir, template_name4)
            assert False, "Expected ValueError for missing 'cookiecutter' key"
        except ValueError as e:
            assert "Context is required to contain a cookiecutter key" in str(e)
        
        # Test 5: Non-existent file should raise FileNotFoundError
        try:
            load(replay_dir, "nonexistent_template")
            assert False, "Expected FileNotFoundError"
        except FileNotFoundError:
            pass
        
        # Test 6: Invalid JSON file should raise JSONDecodeError
        template_name6 = "invalid_json"
        replay_file6 = replay_dir / f"{template_name6}.json"
        with open(replay_file6, 'w', encoding='utf-8') as f:
            f.write("invalid json content")
        
        try:
            load(replay_dir, template_name6)
            assert False, "Expected JSONDecodeError"
        except json.JSONDecodeError:
            pass


# LLM-generated content at query #23
#--------------------------

```python
def test_dump():
    import tempfile
    import json
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = tmpdir
        template_name = "test_template"
        context = {"cookiecutter": {"project_name": "Test Project", "version": "1.0"}}
        
        dump(replay_dir, template_name, context)
        
        expected_file = os.path.join(replay_dir, f"{template_name}.json")
        assert os.path.exists(expected_file)
        
        with open(expected_file, "r", encoding="utf-8") as f:
            loaded_data = json.load(f)
        
        assert loaded_data == context
        
        replay_dir2 = os.path.join(tmpdir, "subdir")
        template_name2 = "template_with_json.json"
        context2 = {"cookiecutter": {"key": "value"}}
        
        dump(replay_dir2, template_name2, context2)
        
        expected_file2 = os.path.join(replay_dir2, template_name2)
        assert os.path.exists(expected_file2)
        
        with open(expected_file2, "r", encoding="utf-8") as f:
            loaded_data2 = json.load(f)
        
        assert loaded_data2 == context2


def test_dump_missing_cookiecutter():
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = tmpdir
        template_name = "test_template"
        context = {"project_name": "Test Project"}
        
        try:
            dump(replay_dir, template_name, context)
            assert False, "Expected ValueError"
        except ValueError as e:
            assert str(e) == "Context is required to contain a cookiecutter key"


def test_dump_nested_structure():
    import tempfile
    import json
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = tmpdir
        template_name = "complex_template"
        context = {
            "cookiecutter": {
                "project": {
                    "name": "Test",
                    "config": {"option1": True, "option2": "value"}
                },
                "list_data": [1, 2, 3],
                "null_value": None
            }
        }
        
        dump(replay_dir, template_name, context)
        
        expected_file = os.path.join(replay_dir, f"{template_name}.json")
        assert os.path.exists(expected_file)
        
        with open(expected_file, "r", encoding="utf-8") as f:
            loaded_data = json.load(f)
        
        assert loaded_data == context


# LLM-generated content at query #24
#--------------------------

```python
def test_dump():
    import tempfile
    import json
    import os
    
    # Test 1: Normal dump with valid context
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = "my_template"
        context = {"cookiecutter": {"project_name": "Test Project", "version": "1.0"}}
        
        dump(tmpdir, template_name, context)
        
        expected_file = os.path.join(tmpdir, f"{template_name}.json")
        assert os.path.exists(expected_file)
        
        with open(expected_file, "r", encoding="utf-8") as f:
            loaded_data = json.load(f)
        
        assert loaded_data == context
    
    # Test 2: Dump with existing .json extension in template_name
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = "my_template.json"
        context = {"cookiecutter": {"key": "value"}}
        
        dump(tmpdir, template_name, context)
        
        expected_file = os.path.join(tmpdir, template_name)
        assert os.path.exists(expected_file)
        
        with open(expected_file, "r", encoding="utf-8") as f:
            loaded_data = json.load(f)
        
        assert loaded_data == context
    
    # Test 3: Dump creates directory if it doesn't exist
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = os.path.join(tmpdir, "subdir", "nested")
        template_name = "test_template"
        context = {"cookiecutter": {"data": "test"}}
        
        dump(replay_dir, template_name, context)
        
        expected_file = os.path.join(replay_dir, f"{template_name}.json")
        assert os.path.exists(expected_file)
    
    # Test 4: Dump raises ValueError when cookiecutter key is missing
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = "bad_template"
        context = {"not_cookiecutter": {"key": "value"}}
        
        try:
            dump(tmpdir, template_name, context)
            assert False, "Expected ValueError"
        except ValueError as e:
            assert "cookiecutter" in str(e)
    
    # Test 5: Dump with nested context structure
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = "complex_template"
        context = {
            "cookiecutter": {
                "project": {
                    "name": "Test",
                    "author": "Developer",
                    "dependencies": ["dep1", "dep2"]
                },
                "choices": {"option1": True, "option2": False}
            }
        }
        
        dump(tmpdir, template_name, context)
        
        expected_file = os.path.join(tmpdir, f"{template_name}.json")
        assert os.path.exists(expected_file)
        
        with open(expected_file, "r", encoding="utf-8") as f:
            loaded_data = json.load(f)
        
        assert loaded_data == context


# LLM-generated content at query #25
#--------------------------

```python
def test_dump():
    import tempfile
    import json
    import os
    
    # Test 1: Normal dump with valid context
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = "my_template"
        context = {"cookiecutter": {"project_name": "Test Project", "version": "1.0"}}
        
        dump(tmpdir, template_name, context)
        
        expected_file = os.path.join(tmpdir, f"{template_name}.json")
        assert os.path.exists(expected_file)
        
        with open(expected_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        assert loaded_data == context
    
    # Test 2: Dump with template name already having .json extension
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = "my_template.json"
        context = {"cookiecutter": {"key": "value"}}
        
        dump(tmpdir, template_name, context)
        
        expected_file = os.path.join(tmpdir, template_name)
        assert os.path.exists(expected_file)
        
        with open(expected_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        assert loaded_data == context
    
    # Test 3: Dump with nested replay directory (should create path)
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = os.path.join(tmpdir, "nested", "dir")
        template_name = "template"
        context = {"cookiecutter": {"test": "data"}}
        
        dump(replay_dir, template_name, context)
        
        expected_file = os.path.join(replay_dir, f"{template_name}.json")
        assert os.path.exists(expected_file)
    
    # Test 4: Dump raises ValueError when context lacks 'cookiecutter' key
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = "bad_template"
        context = {"not_cookiecutter": {"key": "value"}}
        
        try:
            dump(tmpdir, template_name, context)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Context is required to contain a cookiecutter key" in str(e)
    
    # Test 5: Dump with complex nested context
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = "complex_template"
        context = {
            "cookiecutter": {
                "project_name": "My Project",
                "author": "John Doe",
                "features": ["auth", "database", "api"],
                "settings": {
                    "debug": True,
                    "port": 8000
                }
            }
        }
        
        dump(tmpdir, template_name, context)
        
        expected_file = os.path.join(tmpdir, f"{template_name}.json")
        assert os.path.exists(expected_file)
        
        with open(expected_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        assert loaded_data == context
    
    # Test 6: Dump with Path object as replay_dir
    with tempfile.TemporaryDirectory() as tmpdir:
        from pathlib import Path
        replay_dir = Path(tmpdir)
        template_name = "path_template"
        context = {"cookiecutter": {"test": "path_object"}}
        
        dump(replay_dir, template_name, context)
        
        expected_file = os.path.join(tmpdir, f"{template_name}.json")
        assert os.path.exists(expected_file)


# LLM-generated content at query #26
#--------------------------

```python
def test_load():
    import tempfile
    import json
    from pathlib import Path

    # Test normal case with valid context containing cookiecutter key
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "test_template"
        
        # Create valid replay file
        replay_file = replay_dir / f"{template_name}.json"
        valid_context = {"cookiecutter": {"project_name": "Test Project", "version": "1.0"}}
        
        with open(replay_file, "w", encoding="utf-8") as f:
            json.dump(valid_context, f, indent=2)
        
        # Test loading valid context
        result = load(replay_dir, template_name)
        assert result == valid_context
        assert "cookiecutter" in result
        assert result["cookiecutter"]["project_name"] == "Test Project"

    # Test with template name that already has .json extension
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "test_template.json"
        
        replay_file = replay_dir / template_name
        valid_context = {"cookiecutter": {"key": "value"}}
        
        with open(replay_file, "w", encoding="utf-8") as f:
            json.dump(valid_context, f, indent=2)
        
        result = load(replay_dir, template_name)
        assert result == valid_context

    # Test with string path instead of Path object
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = "another_template"
        
        replay_file = Path(tmpdir) / f"{template_name}.json"
        valid_context = {"cookiecutter": {"data": "test"}}
        
        with open(replay_file, "w", encoding="utf-8") as f:
            json.dump(valid_context, f, indent=2)
        
        result = load(tmpdir, template_name)
        assert result == valid_context

    # Test error case: missing cookiecutter key
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "invalid_template"
        
        replay_file = replay_dir / f"{template_name}.json"
        invalid_context = {"not_cookiecutter": {"key": "value"}}
        
        with open(replay_file, "w", encoding="utf-8") as f:
            json.dump(invalid_context, f, indent=2)
        
        try:
            load(replay_dir, template_name)
            assert False, "Expected ValueError for missing cookiecutter key"
        except ValueError as e:
            assert "Context is required to contain a cookiecutter key" in str(e)

    # Test error case: file not found
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "non_existent"
        
        try:
            load(replay_dir, template_name)
            assert False, "Expected FileNotFoundError"
        except FileNotFoundError:
            pass


# LLM-generated content at query #27
#--------------------------

```python
def test_dump(tmp_path):
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {"cookiecutter": {"project_name": "Test Project", "version": "1.0"}}

    dump(replay_dir, template_name, context)

    expected_file = replay_dir / f"{template_name}.json"
    assert expected_file.exists()

    with open(expected_file, encoding="utf-8") as f:
        loaded_data = json.load(f)
    assert loaded_data == context


def test_dump_with_json_extension(tmp_path):
    replay_dir = tmp_path / "replay"
    template_name = "my_template.json"
    context = {"cookiecutter": {"key": "value"}}

    dump(replay_dir, template_name, context)

    expected_file = replay_dir / template_name
    assert expected_file.exists()

    with open(expected_file, encoding="utf-8") as f:
        loaded_data = json.load(f)
    assert loaded_data == context


def test_dump_missing_cookiecutter_key(tmp_path):
    replay_dir = tmp_path / "replay"
    template_name = "template"
    context = {"not_cookiecutter": {"key": "value"}}

    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)


def test_dump_creates_directory(tmp_path):
    replay_dir = tmp_path / "new_dir" / "replay"
    template_name = "template"
    context = {"cookiecutter": {"key": "value"}}

    assert not replay_dir.exists()
    dump(replay_dir, template_name, context)
    assert replay_dir.exists()


def test_dump_with_nested_context(tmp_path):
    replay_dir = tmp_path / "replay"
    template_name = "nested_template"
    context = {
        "cookiecutter": {
            "project": {"name": "Test", "author": "Developer"},
            "settings": {"debug": True, "port": 8080},
        }
    }

    dump(replay_dir, template_name, context)

    expected_file = replay_dir / f"{template_name}.json"
    with open(expected_file, encoding="utf-8") as f:
        loaded_data = json.load(f)
    assert loaded_data == context


# LLM-generated content at query #28
#--------------------------

```python
def test_dump():
    import tempfile
    import json
    import os
    
    # Test 1: Normal dump with valid context
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = "my_template"
        context = {"cookiecutter": {"project_name": "test", "version": "1.0"}}
        
        dump(tmpdir, template_name, context)
        
        expected_file = os.path.join(tmpdir, f"{template_name}.json")
        assert os.path.exists(expected_file)
        
        with open(expected_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        assert loaded_data == context
    
    # Test 2: Dump with template_name already having .json extension
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = "my_template.json"
        context = {"cookiecutter": {"key": "value"}}
        
        dump(tmpdir, template_name, context)
        
        expected_file = os.path.join(tmpdir, template_name)
        assert os.path.exists(expected_file)
        
        with open(expected_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        assert loaded_data == context
    
    # Test 3: Dump creates directory if it doesn't exist
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = os.path.join(tmpdir, "subdir", "another")
        template_name = "test"
        context = {"cookiecutter": {"data": "test"}}
        
        dump(replay_dir, template_name, context)
        
        expected_file = os.path.join(replay_dir, f"{template_name}.json")
        assert os.path.exists(expected_file)
    
    # Test 4: Dump raises ValueError when context doesn't have cookiecutter key
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = "test"
        context = {"not_cookiecutter": {"key": "value"}}
        
        try:
            dump(tmpdir, template_name, context)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "cookiecutter" in str(e)
    
    # Test 5: Dump with nested cookiecutter data
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = "nested"
        context = {
            "cookiecutter": {
                "project": {
                    "name": "test",
                    "author": "test author",
                    "dependencies": ["dep1", "dep2"]
                },
                "choices": {"option": True, "count": 42}
            }
        }
        
        dump(tmpdir, template_name, context)
        
        expected_file = os.path.join(tmpdir, f"{template_name}.json")
        with open(expected_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        assert loaded_data == context


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_load():
    import tempfile
    import json
    from pathlib import Path

    # Test normal case with valid context containing cookiecutter key
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "test_template"
        
        # Create test data
        test_context = {"cookiecutter": {"project_name": "test", "version": "1.0"}}
        replay_file = replay_dir / f"{template_name}.json"
        
        # Write test file
        replay_file.write_text(json.dumps(test_context), encoding="utf-8")
        
        # Test load function
        result = load(replay_dir, template_name)
        assert result == test_context
        assert "cookiecutter" in result

    # Test with template_name already having .json extension
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "test_template.json"
        
        test_context = {"cookiecutter": {"key": "value"}}
        replay_file = replay_dir / template_name
        
        replay_file.write_text(json.dumps(test_context), encoding="utf-8")
        
        result = load(replay_dir, template_name)
        assert result == test_context

    # Test with string path instead of Path object
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = "template"
        
        test_context = {"cookiecutter": {"data": "test"}}
        replay_file = Path(tmpdir) / f"{template_name}.json"
        
        replay_file.write_text(json.dumps(test_context), encoding="utf-8")
        
        result = load(tmpdir, template_name)
        assert result == test_context

    # Test error case: missing cookiecutter key
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "invalid_template"
        
        invalid_context = {"not_cookiecutter": {"key": "value"}}
        replay_file = replay_dir / f"{template_name}.json"
        
        replay_file.write_text(json.dumps(invalid_context), encoding="utf-8")
        
        try:
            load(replay_dir, template_name)
            assert False, "Expected ValueError for missing cookiecutter key"
        except ValueError as e:
            assert "Context is required to contain a cookiecutter key" in str(e)

    # Test error case: file does not exist
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "non_existent"
        
        try:
            load(replay_dir, template_name)
            assert False, "Expected FileNotFoundError"
        except FileNotFoundError:
            pass

    # Test with nested cookiecutter data
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "nested_template"
        
        nested_context = {
            "cookiecutter": {
                "project": {"name": "test", "author": "developer"},
                "settings": {"debug": True, "port": 8000}
            }
        }
        replay_file = replay_dir / f"{template_name}.json"
        
        replay_file.write_text(json.dumps(nested_context), encoding="utf-8")
        
        result = load(replay_dir, template_name)
        assert result == nested_context
        assert "cookiecutter" in result
        assert "project" in result["cookiecutter"]
        assert "settings" in result["cookiecutter"]


# LLM-generated content at query #2
#--------------------------

```python
def test_dump():
    import tempfile
    import json
    import os
    from pathlib import Path
    
    # Test 1: Normal dump with valid context
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "my_template"
        context = {"cookiecutter": {"project_name": "Test Project", "version": "1.0"}}
        
        dump(replay_dir, template_name, context)
        
        expected_file = replay_dir / f"{template_name}.json"
        assert expected_file.exists()
        
        with open(expected_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        assert loaded_data == context
    
    # Test 2: Dump with existing .json extension in template_name
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "my_template.json"
        context = {"cookiecutter": {"key": "value"}}
        
        dump(replay_dir, template_name, context)
        
        expected_file = replay_dir / template_name
        assert expected_file.exists()
        
        with open(expected_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        assert loaded_data == context
    
    # Test 3: Dump creates directory if it doesn't exist
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir) / "subdir" / "nested"
        template_name = "test"
        context = {"cookiecutter": {"data": "test"}}
        
        assert not replay_dir.exists()
        dump(replay_dir, template_name, context)
        assert replay_dir.exists()
    
    # Test 4: Dump raises ValueError when context lacks 'cookiecutter' key
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "test"
        context = {"not_cookiecutter": {"key": "value"}}
        
        try:
            dump(replay_dir, template_name, context)
            assert False, "Expected ValueError"
        except ValueError as e:
            assert "Context is required to contain a cookiecutter key" in str(e)
    
    # Test 5: Dump with string path instead of Path object
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = tmpdir
        template_name = "string_test"
        context = {"cookiecutter": {"test": "data"}}
        
        dump(replay_dir, template_name, context)
        
        expected_file = os.path.join(replay_dir, f"{template_name}.json")
        assert os.path.exists(expected_file)
    
    # Test 6: Dump with complex nested context
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "complex"
        context = {
            "cookiecutter": {
                "project": {
                    "name": "Test",
                    "settings": {"debug": True, "port": 8000}
                },
                "author": "Test Author",
                "dependencies": ["dep1", "dep2"]
            }
        }
        
        dump(replay_dir, template_name, context)
        
        expected_file = replay_dir / f"{template_name}.json"
        with open(expected_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        assert loaded_data == context


# LLM-generated content at query #3
#--------------------------

```python
def test_load():
    import tempfile
    import json
    from pathlib import Path

    # Test normal case with valid cookiecutter context
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "my_template"
        
        # Create test data
        test_context = {"cookiecutter": {"project_name": "Test Project", "version": "1.0"}}
        replay_file = replay_dir / f"{template_name}.json"
        
        # Write test file
        replay_file.write_text(json.dumps(test_context), encoding="utf-8")
        
        # Test load function
        result = load(replay_dir, template_name)
        assert result == test_context
        assert "cookiecutter" in result

    # Test with template name that already has .json extension
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "my_template.json"
        
        # Create test data
        test_context = {"cookiecutter": {"key": "value"}}
        replay_file = replay_dir / template_name
        
        # Write test file
        replay_file.write_text(json.dumps(test_context), encoding="utf-8")
        
        # Test load function
        result = load(replay_dir, template_name)
        assert result == test_context

    # Test with string path instead of Path object
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = "template"
        
        # Create test data
        test_context = {"cookiecutter": {"data": "test"}}
        replay_file = Path(tmpdir) / f"{template_name}.json"
        
        # Write test file
        replay_file.write_text(json.dumps(test_context), encoding="utf-8")
        
        # Test load function with string path
        result = load(tmpdir, template_name)
        assert result == test_context

    # Test error case: missing cookiecutter key
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "invalid_template"
        
        # Create invalid test data (missing cookiecutter key)
        invalid_context = {"project_name": "Test"}
        replay_file = replay_dir / f"{template_name}.json"
        
        # Write test file
        replay_file.write_text(json.dumps(invalid_context), encoding="utf-8")
        
        # Test that ValueError is raised
        try:
            load(replay_dir, template_name)
            assert False, "Expected ValueError for missing cookiecutter key"
        except ValueError as e:
            assert "cookiecutter" in str(e)

    # Test error case: file not found
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "non_existent"
        
        # Test that FileNotFoundError is raised
        try:
            load(replay_dir, template_name)
            assert False, "Expected FileNotFoundError"
        except FileNotFoundError:
            pass


# LLM-generated content at query #4
#--------------------------

```python
def test_dump():
    import tempfile
    import json
    import os
    from pathlib import Path
    
    # Test 1: Normal dump with valid context
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "test_template"
        context = {"cookiecutter": {"project_name": "Test Project", "version": "1.0"}}
        
        dump(replay_dir, template_name, context)
        
        expected_file = replay_dir / "test_template.json"
        assert expected_file.exists()
        
        with open(expected_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        assert loaded_data == context
    
    # Test 2: Dump with .json suffix in template_name
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "test_template.json"
        context = {"cookiecutter": {"key": "value"}}
        
        dump(replay_dir, template_name, context)
        
        expected_file = replay_dir / "test_template.json"
        assert expected_file.exists()
        
        with open(expected_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        assert loaded_data == context
    
    # Test 3: Dump with string path instead of Path object
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = "another_template"
        context = {"cookiecutter": {"name": "Test"}}
        
        dump(tmpdir, template_name, context)
        
        expected_file = os.path.join(tmpdir, "another_template.json")
        assert os.path.exists(expected_file)
        
        with open(expected_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        assert loaded_data == context
    
    # Test 4: Raises ValueError when context doesn't contain 'cookiecutter' key
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "bad_template"
        context = {"not_cookiecutter": {"key": "value"}}
        
        try:
            dump(replay_dir, template_name, context)
            assert False, "Expected ValueError"
        except ValueError as e:
            assert str(e) == "Context is required to contain a cookiecutter key"
    
    # Test 5: Creates directory if it doesn't exist
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir) / "subdir" / "nested"
        template_name = "nested_template"
        context = {"cookiecutter": {"test": "data"}}
        
        assert not replay_dir.exists()
        dump(replay_dir, template_name, context)
        
        expected_file = replay_dir / "nested_template.json"
        assert expected_file.exists()
        
        with open(expected_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        assert loaded_data == context


# LLM-generated content at query #5
#--------------------------

```python
def test_load():
    import tempfile
    import json
    from pathlib import Path

    # Test normal case with valid JSON containing cookiecutter key
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "test_template"
        
        # Create test data
        test_context = {"cookiecutter": {"project_name": "Test Project", "version": "1.0"}}
        
        # Write test file
        replay_file = replay_dir / f"{template_name}.json"
        with open(replay_file, "w", encoding="utf-8") as f:
            json.dump(test_context, f, indent=2)
        
        # Test load function
        result = load(replay_dir, template_name)
        assert result == test_context
        assert "cookiecutter" in result

    # Test with template_name that already has .json extension
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "test_template.json"
        
        test_context = {"cookiecutter": {"key": "value"}}
        
        replay_file = replay_dir / template_name
        with open(replay_file, "w", encoding="utf-8") as f:
            json.dump(test_context, f, indent=2)
        
        result = load(replay_dir, template_name)
        assert result == test_context

    # Test with string path instead of Path object
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = "another_template"
        
        test_context = {"cookiecutter": {"data": "test"}}
        
        replay_file = Path(tmpdir) / f"{template_name}.json"
        with open(replay_file, "w", encoding="utf-8") as f:
            json.dump(test_context, f, indent=2)
        
        result = load(tmpdir, template_name)
        assert result == test_context

    # Test error case: missing cookiecutter key
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "invalid_template"
        
        invalid_context = {"not_cookiecutter": {"key": "value"}}
        
        replay_file = replay_dir / f"{template_name}.json"
        with open(replay_file, "w", encoding="utf-8") as f:
            json.dump(invalid_context, f, indent=2)
        
        try:
            load(replay_dir, template_name)
            assert False, "Expected ValueError for missing cookiecutter key"
        except ValueError as e:
            assert "Context is required to contain a cookiecutter key" in str(e)

    # Test error case: file not found
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "non_existent"
        
        try:
            load(replay_dir, template_name)
            assert False, "Expected FileNotFoundError"
        except FileNotFoundError:
            pass

    # Test with nested cookiecutter data
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "nested_template"
        
        nested_context = {
            "cookiecutter": {
                "project": {
                    "name": "Test",
                    "author": "Developer"
                },
                "settings": {
                    "debug": True,
                    "port": 8000
                }
            }
        }
        
        replay_file = replay_dir / f"{template_name}.json"
        with open(replay_file, "w", encoding="utf-8") as f:
            json.dump(nested_context, f, indent=2)
        
        result = load(replay_dir, template_name)
        assert result == nested_context
        assert "cookiecutter" in result
        assert "project" in result["cookiecutter"]
        assert "settings" in result["cookiecutter"]


# LLM-generated content at query #6
#--------------------------

```python
def test_load():
    import tempfile
    import json
    from pathlib import Path

    # Test normal case with valid context containing cookiecutter key
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "test_template"
        
        # Create test data
        test_context = {
            "cookiecutter": {
                "project_name": "Test Project",
                "author": "Test Author"
            },
            "other_key": "other_value"
        }
        
        # Write test file
        replay_file = replay_dir / f"{template_name}.json"
        with open(replay_file, 'w', encoding='utf-8') as f:
            json.dump(test_context, f)
        
        # Test load function
        result = load(replay_dir, template_name)
        assert result == test_context
        assert "cookiecutter" in result
        assert result["cookiecutter"]["project_name"] == "Test Project"

    # Test with template_name already having .json extension
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "test_template.json"
        
        test_context = {
            "cookiecutter": {
                "key": "value"
            }
        }
        
        replay_file = replay_dir / template_name
        with open(replay_file, 'w', encoding='utf-8') as f:
            json.dump(test_context, f)
        
        result = load(replay_dir, template_name)
        assert result == test_context

    # Test with string path instead of Path object
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = "test_template"
        
        test_context = {
            "cookiecutter": {
                "key": "value"
            }
        }
        
        replay_file = Path(tmpdir) / f"{template_name}.json"
        with open(replay_file, 'w', encoding='utf-8') as f:
            json.dump(test_context, f)
        
        result = load(tmpdir, template_name)
        assert result == test_context

    # Test error case: missing cookiecutter key
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "invalid_template"
        
        invalid_context = {
            "not_cookiecutter": {
                "key": "value"
            }
        }
        
        replay_file = replay_dir / f"{template_name}.json"
        with open(replay_file, 'w', encoding='utf-8') as f:
            json.dump(invalid_context, f)
        
        try:
            load(replay_dir, template_name)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Context is required to contain a cookiecutter key" in str(e)

    # Test error case: file not found
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "non_existent_template"
        
        try:
            load(replay_dir, template_name)
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError:
            pass

    # Test with nested cookiecutter structure
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "nested_template"
        
        nested_context = {
            "cookiecutter": {
                "nested": {
                    "deep": {
                        "value": "test"
                    }
                }
            }
        }
        
        replay_file = replay_dir / f"{template_name}.json"
        with open(replay_file, 'w', encoding='utf-8') as f:
            json.dump(nested_context, f)
        
        result = load(replay_dir, template_name)
        assert result == nested_context
        assert result["cookiecutter"]["nested"]["deep"]["value"] == "test"


# LLM-generated content at query #7
#--------------------------

```python
def test_load():
    # Test normal case with valid JSON containing cookiecutter key
    import tempfile
    import json
    from pathlib import Path
    
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "test_template"
        
        # Create test JSON file
        test_context = {"cookiecutter": {"project_name": "Test Project", "version": "1.0"}}
        replay_file = replay_dir / f"{template_name}.json"
        
        with open(replay_file, "w", encoding="utf-8") as f:
            json.dump(test_context, f)
        
        # Test loading
        result = load(replay_dir, template_name)
        assert result == test_context
        assert "cookiecutter" in result
        assert result["cookiecutter"]["project_name"] == "Test Project"
    
    # Test with template_name that already has .json extension
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "test_template.json"
        
        test_context = {"cookiecutter": {"key": "value"}}
        replay_file = replay_dir / template_name
        
        with open(replay_file, "w", encoding="utf-8") as f:
            json.dump(test_context, f)
        
        result = load(replay_dir, template_name)
        assert result == test_context
    
    # Test with string path instead of Path object
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = "another_template"
        
        test_context = {"cookiecutter": {"data": "test"}}
        replay_file = os.path.join(tmpdir, f"{template_name}.json")
        
        with open(replay_file, "w", encoding="utf-8") as f:
            json.dump(test_context, f)
        
        result = load(tmpdir, template_name)
        assert result == test_context
    
    # Test error case: missing cookiecutter key
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "bad_template"
        
        bad_context = {"not_cookiecutter": {"key": "value"}}
        replay_file = replay_dir / f"{template_name}.json"
        
        with open(replay_file, "w", encoding="utf-8") as f:
            json.dump(bad_context, f)
        
        try:
            load(replay_dir, template_name)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "cookiecutter" in str(e)
    
    # Test error case: file not found
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "nonexistent"
        
        try:
            load(replay_dir, template_name)
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError:
            pass
    
    # Test with nested cookiecutter structure
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "complex_template"
        
        complex_context = {
            "cookiecutter": {
                "project": {
                    "name": "My Project",
                    "author": "Test Author"
                },
                "settings": {
                    "debug": True,
                    "version": "2.0"
                }
            }
        }
        replay_file = replay_dir / f"{template_name}.json"
        
        with open(replay_file, "w", encoding="utf-8") as f:
            json.dump(complex_context, f)
        
        result = load(replay_dir, template_name)
        assert result == complex_context
        assert result["cookiecutter"]["project"]["name"] == "My Project"
        assert result["cookiecutter"]["settings"]["debug"] is True


# LLM-generated content at query #8
#--------------------------

```python
def test_dump():
    import tempfile
    import json
    import os
    
    # Test 1: Normal dump with valid context
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = "my_template"
        context = {"cookiecutter": {"project_name": "Test Project", "version": "1.0"}}
        
        dump(tmpdir, template_name, context)
        
        expected_file = os.path.join(tmpdir, f"{template_name}.json")
        assert os.path.exists(expected_file)
        
        with open(expected_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        assert loaded_data == context
    
    # Test 2: Dump with template name ending in .json
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = "my_template.json"
        context = {"cookiecutter": {"key": "value"}}
        
        dump(tmpdir, template_name, context)
        
        expected_file = os.path.join(tmpdir, template_name)
        assert os.path.exists(expected_file)
        
        with open(expected_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        assert loaded_data == context
    
    # Test 3: Dump creates directory if it doesn't exist
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = os.path.join(tmpdir, "subdir", "nested")
        template_name = "test_template"
        context = {"cookiecutter": {"data": "test"}}
        
        dump(replay_dir, template_name, context)
        
        expected_file = os.path.join(replay_dir, f"{template_name}.json")
        assert os.path.exists(expected_file)
    
    # Test 4: Dump raises ValueError when cookiecutter key is missing
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = "bad_template"
        context = {"not_cookiecutter": {"key": "value"}}
        
        try:
            dump(tmpdir, template_name, context)
            assert False, "Expected ValueError to be raised"
        except ValueError as e:
            assert "cookiecutter" in str(e)
    
    # Test 5: Dump with complex nested context
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = "complex_template"
        context = {
            "cookiecutter": {
                "project": {
                    "name": "Test",
                    "settings": {"debug": True, "port": 8000}
                },
                "author": "Test Author",
                "dependencies": ["dep1", "dep2"]
            }
        }
        
        dump(tmpdir, template_name, context)
        
        expected_file = os.path.join(tmpdir, f"{template_name}.json")
        assert os.path.exists(expected_file)
        
        with open(expected_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        assert loaded_data == context


# LLM-generated content at query #9
#--------------------------

```python
def test_dump():
    import tempfile
    import json
    import os
    from pathlib import Path
    
    # Test 1: Normal dump with valid context
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "my_template"
        context = {"cookiecutter": {"project_name": "Test Project", "version": "1.0"}}
        
        dump(replay_dir, template_name, context)
        
        expected_file = replay_dir / f"{template_name}.json"
        assert expected_file.exists()
        
        with open(expected_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        assert loaded_data == context
    
    # Test 2: Dump with existing .json extension in template_name
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "my_template.json"
        context = {"cookiecutter": {"key": "value"}}
        
        dump(replay_dir, template_name, context)
        
        expected_file = replay_dir / template_name
        assert expected_file.exists()
        
        with open(expected_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        assert loaded_data == context
    
    # Test 3: Dump creates directory if it doesn't exist
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir) / "subdir" / "nested"
        template_name = "test"
        context = {"cookiecutter": {"data": "test"}}
        
        assert not replay_dir.exists()
        dump(replay_dir, template_name, context)
        assert replay_dir.exists()
    
    # Test 4: Dump raises ValueError when context lacks 'cookiecutter' key
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "test"
        context = {"not_cookiecutter": {"key": "value"}}
        
        try:
            dump(replay_dir, template_name, context)
            assert False, "Expected ValueError"
        except ValueError as e:
            assert str(e) == "Context is required to contain a cookiecutter key"
    
    # Test 5: Dump with complex nested context
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "complex"
        context = {
            "cookiecutter": {
                "project": {
                    "name": "Test",
                    "author": "Developer",
                    "dependencies": ["dep1", "dep2"]
                },
                "settings": {"debug": True, "port": 8080}
            }
        }
        
        dump(replay_dir, template_name, context)
        
        expected_file = replay_dir / f"{template_name}.json"
        with open(expected_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        assert loaded_data == context
    
    # Test 6: Dump with string replay_dir instead of Path
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = "string_dir"
        context = {"cookiecutter": {"test": "value"}}
        
        dump(tmpdir, template_name, context)
        
        expected_file = os.path.join(tmpdir, f"{template_name}.json")
        assert os.path.exists(expected_file)


# LLM-generated content at query #10
#--------------------------

```python
def test_dump():
    import tempfile
    import json
    import os
    from pathlib import Path

    # Test 1: Normal dump with valid context
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "test_template"
        context = {"cookiecutter": {"project_name": "Test Project", "version": "1.0"}}
        
        dump(replay_dir, template_name, context)
        
        expected_file = replay_dir / "test_template.json"
        assert expected_file.exists()
        
        with open(expected_file, "r", encoding="utf-8") as f:
            loaded_context = json.load(f)
        
        assert loaded_context == context

    # Test 2: Dump with template_name already having .json extension
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "test_template.json"
        context = {"cookiecutter": {"key": "value"}}
        
        dump(replay_dir, template_name, context)
        
        expected_file = replay_dir / "test_template.json"
        assert expected_file.exists()

    # Test 3: Dump with string path instead of Path object
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = "template"
        context = {"cookiecutter": {"data": "test"}}
        
        dump(tmpdir, template_name, context)
        
        expected_file = os.path.join(tmpdir, "template.json")
        assert os.path.exists(expected_file)

    # Test 4: Dump creates directory if it doesn't exist
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir) / "subdir" / "nested"
        template_name = "test"
        context = {"cookiecutter": {"test": "value"}}
        
        dump(replay_dir, template_name, context)
        
        expected_file = replay_dir / "test.json"
        assert expected_file.exists()

    # Test 5: Dump raises ValueError when context doesn't contain 'cookiecutter' key
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "test"
        context = {"not_cookiecutter": {"key": "value"}}
        
        try:
            dump(replay_dir, template_name, context)
            assert False, "Expected ValueError"
        except ValueError as e:
            assert "Context is required to contain a cookiecutter key" in str(e)

    # Test 6: Dump with complex nested context
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "complex"
        context = {
            "cookiecutter": {
                "project": {
                    "name": "Test",
                    "settings": {"debug": True, "port": 8000}
                },
                "list_data": [1, 2, 3],
                "null_value": None
            }
        }
        
        dump(replay_dir, template_name, context)
        
        expected_file = replay_dir / "complex.json"
        with open(expected_file, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        
        assert loaded == context


# LLM-generated content at query #11
#--------------------------

```python
def test_load():
    import tempfile
    import json
    from pathlib import Path

    # Test normal case with valid context containing cookiecutter key
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "test_template"
        
        # Create valid replay file
        replay_file = replay_dir / f"{template_name}.json"
        valid_context = {"cookiecutter": {"project_name": "Test Project", "version": "1.0"}}
        
        with open(replay_file, "w", encoding="utf-8") as f:
            json.dump(valid_context, f, indent=2)
        
        # Test loading valid file
        result = load(replay_dir, template_name)
        assert result == valid_context
        assert "cookiecutter" in result
        assert result["cookiecutter"]["project_name"] == "Test Project"

    # Test with template name that already has .json extension
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "test_template.json"
        
        replay_file = replay_dir / template_name
        valid_context = {"cookiecutter": {"key": "value"}}
        
        with open(replay_file, "w", encoding="utf-8") as f:
            json.dump(valid_context, f, indent=2)
        
        result = load(replay_dir, template_name)
        assert result == valid_context

    # Test with string path instead of Path object
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = "another_template"
        
        replay_file = Path(tmpdir) / f"{template_name}.json"
        valid_context = {"cookiecutter": {"data": "test"}}
        
        with open(replay_file, "w", encoding="utf-8") as f:
            json.dump(valid_context, f, indent=2)
        
        result = load(tmpdir, template_name)
        assert result == valid_context

    # Test error case: missing cookiecutter key
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "invalid_template"
        
        replay_file = replay_dir / f"{template_name}.json"
        invalid_context = {"not_cookiecutter": {"key": "value"}}
        
        with open(replay_file, "w", encoding="utf-8") as f:
            json.dump(invalid_context, f, indent=2)
        
        try:
            load(replay_dir, template_name)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Context is required to contain a cookiecutter key" in str(e)

    # Test error case: file doesn't exist
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "non_existent"
        
        try:
            load(replay_dir, template_name)
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError:
            pass

    # Test with nested cookiecutter structure
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "nested_template"
        
        replay_file = replay_dir / f"{template_name}.json"
        nested_context = {
            "cookiecutter": {
                "project": {
                    "name": "Test",
                    "author": "Developer"
                },
                "settings": {
                    "debug": True,
                    "port": 8000
                }
            }
        }
        
        with open(replay_file, "w", encoding="utf-8") as f:
            json.dump(nested_context, f, indent=2)
        
        result = load(replay_dir, template_name)
        assert result == nested_context
        assert result["cookiecutter"]["project"]["name"] == "Test"
        assert result["cookiecutter"]["settings"]["debug"] is True


# LLM-generated content at query #12
#--------------------------

```python
def test_dump():
    import tempfile
    import json
    import os
    
    # Test 1: Normal dump with valid context
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = "my_template"
        context = {"cookiecutter": {"project_name": "Test Project", "version": "1.0"}}
        
        dump(tmpdir, template_name, context)
        
        expected_file = os.path.join(tmpdir, f"{template_name}.json")
        assert os.path.exists(expected_file)
        
        with open(expected_file, "r", encoding="utf-8") as f:
            loaded_context = json.load(f)
        
        assert loaded_context == context
    
    # Test 2: Dump with existing .json extension in template_name
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = "my_template.json"
        context = {"cookiecutter": {"key": "value"}}
        
        dump(tmpdir, template_name, context)
        
        expected_file = os.path.join(tmpdir, template_name)
        assert os.path.exists(expected_file)
        
        with open(expected_file, "r", encoding="utf-8") as f:
            loaded_context = json.load(f)
        
        assert loaded_context == context
    
    # Test 3: Dump with nested replay_dir path
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = os.path.join(tmpdir, "nested", "dir")
        template_name = "template"
        context = {"cookiecutter": {"data": "test"}}
        
        dump(replay_dir, template_name, context)
        
        expected_file = os.path.join(replay_dir, f"{template_name}.json")
        assert os.path.exists(expected_file)
        
        with open(expected_file, "r", encoding="utf-8") as f:
            loaded_context = json.load(f)
        
        assert loaded_context == context
    
    # Test 4: Dump raises ValueError when cookiecutter key is missing
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = "test"
        context = {"not_cookiecutter": {"key": "value"}}
        
        try:
            dump(tmpdir, template_name, context)
            assert False, "Expected ValueError"
        except ValueError as e:
            assert "Context is required to contain a cookiecutter key" in str(e)
    
    # Test 5: Dump with Path object as replay_dir
    with tempfile.TemporaryDirectory() as tmpdir:
        from pathlib import Path
        replay_dir = Path(tmpdir)
        template_name = "path_template"
        context = {"cookiecutter": {"test": "path"}}
        
        dump(replay_dir, template_name, context)
        
        expected_file = os.path.join(tmpdir, f"{template_name}.json")
        assert os.path.exists(expected_file)
        
        with open(expected_file, "r", encoding="utf-8") as f:
            loaded_context = json.load(f)
        
        assert loaded_context == context


# LLM-generated content at query #13
#--------------------------

```python
def test_load():
    import tempfile
    import json
    from pathlib import Path

    # Test 1: Normal load with valid context containing cookiecutter key
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "test_template"
        
        # Create valid replay file
        valid_context = {"cookiecutter": {"project_name": "Test Project", "version": "1.0"}}
        replay_file = replay_dir / f"{template_name}.json"
        
        with open(replay_file, "w", encoding="utf-8") as f:
            json.dump(valid_context, f, indent=2)
        
        result = load(replay_dir, template_name)
        assert result == valid_context
        assert "cookiecutter" in result

    # Test 2: Load with template_name already having .json extension
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "test_template.json"
        
        valid_context = {"cookiecutter": {"key": "value"}}
        replay_file = replay_dir / template_name
        
        with open(replay_file, "w", encoding="utf-8") as f:
            json.dump(valid_context, f, indent=2)
        
        result = load(replay_dir, template_name)
        assert result == valid_context

    # Test 3: Load with replay_dir as string instead of Path
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = "string_test"
        
        valid_context = {"cookiecutter": {"data": "test"}}
        replay_file = Path(tmpdir) / f"{template_name}.json"
        
        with open(replay_file, "w", encoding="utf-8") as f:
            json.dump(valid_context, f, indent=2)
        
        result = load(tmpdir, template_name)
        assert result == valid_context

    # Test 4: Raise ValueError when cookiecutter key is missing
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "invalid_template"
        
        invalid_context = {"not_cookiecutter": {"key": "value"}}
        replay_file = replay_dir / f"{template_name}.json"
        
        with open(replay_file, "w", encoding="utf-8") as f:
            json.dump(invalid_context, f, indent=2)
        
        try:
            load(replay_dir, template_name)
            assert False, "Expected ValueError"
        except ValueError as e:
            assert "Context is required to contain a cookiecutter key" in str(e)

    # Test 5: File not found should raise FileNotFoundError
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "non_existent"
        
        try:
            load(replay_dir, template_name)
            assert False, "Expected FileNotFoundError"
        except FileNotFoundError:
            pass

    # Test 6: Invalid JSON should raise JSONDecodeError
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "invalid_json"
        
        replay_file = replay_dir / f"{template_name}.json"
        
        with open(replay_file, "w", encoding="utf-8") as f:
            f.write("invalid json content")
        
        try:
            load(replay_dir, template_name)
            assert False, "Expected JSONDecodeError"
        except json.JSONDecodeError:
            pass


# LLM-generated content at query #14
#--------------------------

```python
def test_load():
    import tempfile
    import json
    from pathlib import Path

    # Test normal case with valid context containing cookiecutter key
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "test_template"
        
        # Create a valid context file
        context = {"cookiecutter": {"project_name": "Test Project", "version": "1.0"}}
        replay_file = replay_dir / f"{template_name}.json"
        
        with open(replay_file, "w", encoding="utf-8") as f:
            json.dump(context, f)
        
        # Test loading the file
        result = load(replay_dir, template_name)
        assert result == context
        assert "cookiecutter" in result
        assert result["cookiecutter"]["project_name"] == "Test Project"

    # Test with template name that already has .json extension
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "test_template.json"
        
        context = {"cookiecutter": {"key": "value"}}
        replay_file = replay_dir / template_name
        
        with open(replay_file, "w", encoding="utf-8") as f:
            json.dump(context, f)
        
        result = load(replay_dir, template_name)
        assert result == context

    # Test with string path instead of Path object
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = "template"
        
        context = {"cookiecutter": {"data": "test"}}
        replay_file = Path(tmpdir) / f"{template_name}.json"
        
        with open(replay_file, "w", encoding="utf-8") as f:
            json.dump(context, f)
        
        result = load(tmpdir, template_name)
        assert result == context

    # Test error case: missing cookiecutter key
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "invalid_template"
        
        # Create context without cookiecutter key
        context = {"project": "Test"}
        replay_file = replay_dir / f"{template_name}.json"
        
        with open(replay_file, "w", encoding="utf-8") as f:
            json.dump(context, f)
        
        try:
            load(replay_dir, template_name)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Context is required to contain a cookiecutter key" in str(e)

    # Test error case: file doesn't exist
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "nonexistent"
        
        try:
            load(replay_dir, template_name)
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError:
            pass  # Expected

    # Test with nested cookiecutter data
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "complex_template"
        
        context = {
            "cookiecutter": {
                "project": {
                    "name": "Test",
                    "author": "Developer"
                },
                "settings": ["option1", "option2"]
            }
        }
        replay_file = replay_dir / f"{template_name}.json"
        
        with open(replay_file, "w", encoding="utf-8") as f:
            json.dump(context, f)
        
        result = load(replay_dir, template_name)
        assert result == context
        assert isinstance(result["cookiecutter"]["project"], dict)
        assert isinstance(result["cookiecutter"]["settings"], list)


# LLM-generated content at query #15
#--------------------------

```python
def test_dump():
    import tempfile
    import json
    import os
    
    # Test 1: Normal dump with valid context
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = "my_template"
        context = {"cookiecutter": {"project_name": "Test Project", "version": "1.0"}}
        
        dump(tmpdir, template_name, context)
        
        expected_file = os.path.join(tmpdir, f"{template_name}.json")
        assert os.path.exists(expected_file)
        
        with open(expected_file, "r", encoding="utf-8") as f:
            loaded_context = json.load(f)
        
        assert loaded_context == context
    
    # Test 2: Dump with existing .json extension in template_name
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = "my_template.json"
        context = {"cookiecutter": {"key": "value"}}
        
        dump(tmpdir, template_name, context)
        
        expected_file = os.path.join(tmpdir, template_name)
        assert os.path.exists(expected_file)
        
        with open(expected_file, "r", encoding="utf-8") as f:
            loaded_context = json.load(f)
        
        assert loaded_context == context
    
    # Test 3: Dump creates directory if it doesn't exist
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = os.path.join(tmpdir, "subdir", "nested")
        template_name = "test"
        context = {"cookiecutter": {"data": "test"}}
        
        dump(replay_dir, template_name, context)
        
        expected_file = os.path.join(replay_dir, f"{template_name}.json")
        assert os.path.exists(expected_file)
    
    # Test 4: Dump raises ValueError when context lacks 'cookiecutter' key
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = "test"
        context = {"not_cookiecutter": {"key": "value"}}
        
        try:
            dump(tmpdir, template_name, context)
            assert False, "Expected ValueError"
        except ValueError as e:
            assert "Context is required to contain a cookiecutter key" in str(e)
    
    # Test 5: Dump with complex nested context
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = "complex"
        context = {
            "cookiecutter": {
                "project": {
                    "name": "Test",
                    "authors": ["Alice", "Bob"],
                    "settings": {"debug": True, "port": 8080}
                },
                "choices": [1, 2, 3],
                "description": "A test project"
            }
        }
        
        dump(tmpdir, template_name, context)
        
        expected_file = os.path.join(tmpdir, f"{template_name}.json")
        with open(expected_file, "r", encoding="utf-8") as f:
            loaded_context = json.load(f)
        
        assert loaded_context == context


# LLM-generated content at query #16
#--------------------------

```python
def test_load():
    import json
    import tempfile
    from pathlib import Path
    
    # Test normal case with valid context containing cookiecutter key
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "test_template"
        
        # Create valid context data
        test_context = {
            "cookiecutter": {
                "project_name": "Test Project",
                "author": "Test Author",
                "version": "1.0.0"
            },
            "other_data": "some_value"
        }
        
        # Write the test file
        replay_file = replay_dir / f"{template_name}.json"
        with open(replay_file, "w", encoding="utf-8") as f:
            json.dump(test_context, f, indent=2)
        
        # Test loading the file
        result = load(replay_dir, template_name)
        assert result == test_context
        assert "cookiecutter" in result
        assert result["cookiecutter"]["project_name"] == "Test Project"
    
    # Test case where template_name already has .json extension
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "test_template.json"
        
        test_context = {
            "cookiecutter": {"key": "value"}
        }
        
        replay_file = replay_dir / template_name
        with open(replay_file, "w", encoding="utf-8") as f:
            json.dump(test_context, f, indent=2)
        
        result = load(replay_dir, template_name)
        assert result == test_context
    
    # Test case where context doesn't contain cookiecutter key
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "invalid_template"
        
        invalid_context = {
            "not_cookiecutter": {"key": "value"}
        }
        
        replay_file = replay_dir / f"{template_name}.json"
        with open(replay_file, "w", encoding="utf-8") as f:
            json.dump(invalid_context, f, indent=2)
        
        try:
            load(replay_dir, template_name)
            assert False, "Expected ValueError for missing cookiecutter key"
        except ValueError as e:
            assert "Context is required to contain a cookiecutter key" in str(e)
    
    # Test with string path instead of Path object
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = "string_path_test"
        
        test_context = {
            "cookiecutter": {"test": "data"}
        }
        
        replay_file = Path(tmpdir) / f"{template_name}.json"
        with open(replay_file, "w", encoding="utf-8") as f:
            json.dump(test_context, f, indent=2)
        
        result = load(tmpdir, template_name)
        assert result == test_context
    
    # Test with nested data in cookiecutter
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "nested_data"
        
        nested_context = {
            "cookiecutter": {
                "project": {
                    "name": "Test",
                    "config": {
                        "debug": True,
                        "port": 8080
                    }
                },
                "list_data": [1, 2, 3]
            }
        }
        
        replay_file = replay_dir / f"{template_name}.json"
        with open(replay_file, "w", encoding="utf-8") as f:
            json.dump(nested_context, f, indent=2)
        
        result = load(replay_dir, template_name)
        assert result == nested_context
        assert result["cookiecutter"]["project"]["config"]["debug"] is True
        assert result["cookiecutter"]["list_data"] == [1, 2, 3]


# LLM-generated content at query #17
#--------------------------

```python
def test_load():
    import tempfile
    import json
    from pathlib import Path

    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "test_template"
        
        # Test 1: Normal case with valid context containing 'cookiecutter' key
        test_context = {
            "cookiecutter": {
                "project_name": "Test Project",
                "author": "Test Author",
                "version": "1.0.0"
            }
        }
        
        # Create the replay file
        replay_file = replay_dir / f"{template_name}.json"
        with open(replay_file, 'w', encoding='utf-8') as f:
            json.dump(test_context, f, indent=2)
        
        # Load the context
        loaded_context = load(replay_dir, template_name)
        
        # Verify the loaded context matches the original
        assert loaded_context == test_context
        assert 'cookiecutter' in loaded_context
        assert loaded_context['cookiecutter']['project_name'] == "Test Project"
        
        # Test 2: Template name without .json extension
        template_name2 = "another_template"
        test_context2 = {
            "cookiecutter": {
                "key": "value"
            }
        }
        
        replay_file2 = replay_dir / f"{template_name2}.json"
        with open(replay_file2, 'w', encoding='utf-8') as f:
            json.dump(test_context2, f, indent=2)
        
        loaded_context2 = load(replay_dir, template_name2)
        assert loaded_context2 == test_context2
        
        # Test 3: Template name with .json extension
        template_name3 = "template_with_ext.json"
        test_context3 = {
            "cookiecutter": {
                "data": "test_data"
            }
        }
        
        replay_file3 = replay_dir / template_name3
        with open(replay_file3, 'w', encoding='utf-8') as f:
            json.dump(test_context3, f, indent=2)
        
        loaded_context3 = load(replay_dir, template_name3)
        assert loaded_context3 == test_context3
        
        # Test 4: Context missing 'cookiecutter' key should raise ValueError
        template_name4 = "bad_template"
        bad_context = {
            "not_cookiecutter": {
                "key": "value"
            }
        }
        
        replay_file4 = replay_dir / f"{template_name4}.json"
        with open(replay_file4, 'w', encoding='utf-8') as f:
            json.dump(bad_context, f, indent=2)
        
        try:
            load(replay_dir, template_name4)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Context is required to contain a cookiecutter key" in str(e)
        
        # Test 5: Non-existent file should raise FileNotFoundError
        try:
            load(replay_dir, "non_existent_template")
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError:
            pass
        
        # Test 6: Invalid JSON file should raise JSONDecodeError
        template_name6 = "invalid_json"
        replay_file6 = replay_dir / f"{template_name6}.json"
        with open(replay_file6, 'w', encoding='utf-8') as f:
            f.write("invalid json content")
        
        try:
            load(replay_dir, template_name6)
            assert False, "Should have raised JSONDecodeError"
        except json.JSONDecodeError:
            pass


# LLM-generated content at query #18
#--------------------------

```python
def test_load():
    import tempfile
    import json
    from pathlib import Path

    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "test_template"
        
        # Test 1: Normal case with valid context containing 'cookiecutter' key
        test_context = {"cookiecutter": {"project_name": "Test Project", "version": "1.0"}}
        replay_file = replay_dir / f"{template_name}.json"
        
        # Write test file
        replay_file.write_text(json.dumps(test_context), encoding="utf-8")
        
        # Load and verify
        result = load(replay_dir, template_name)
        assert result == test_context
        assert "cookiecutter" in result
        
        # Test 2: Template name without .json extension
        test_context2 = {"cookiecutter": {"key": "value"}}
        replay_file2 = replay_dir / "template2.json"
        replay_file2.write_text(json.dumps(test_context2), encoding="utf-8")
        
        result = load(replay_dir, "template2")
        assert result == test_context2
        
        # Test 3: Template name with .json extension
        test_context3 = {"cookiecutter": {"data": "test"}}
        replay_file3 = replay_dir / "template3.json"
        replay_file3.write_text(json.dumps(test_context3), encoding="utf-8")
        
        result = load(replay_dir, "template3.json")
        assert result == test_context3
        
        # Test 4: Missing cookiecutter key should raise ValueError
        invalid_context = {"not_cookiecutter": {"key": "value"}}
        replay_file4 = replay_dir / "invalid.json"
        replay_file4.write_text(json.dumps(invalid_context), encoding="utf-8")
        
        try:
            load(replay_dir, "invalid")
            assert False, "Expected ValueError for missing cookiecutter key"
        except ValueError as e:
            assert "Context is required to contain a cookiecutter key" in str(e)
        
        # Test 5: Non-existent file should raise FileNotFoundError
        try:
            load(replay_dir, "nonexistent")
            assert False, "Expected FileNotFoundError"
        except FileNotFoundError:
            pass
        
        # Test 6: Invalid JSON should raise JSONDecodeError
        replay_file5 = replay_dir / "badjson.json"
        replay_file5.write_text("invalid json content", encoding="utf-8")
        
        try:
            load(replay_dir, "badjson")
            assert False, "Expected JSONDecodeError"
        except json.JSONDecodeError:
            pass


# LLM-generated content at query #19
#--------------------------

```python
def test_load():
    import tempfile
    import json
    from pathlib import Path
    
    # Test 1: Normal load with valid context containing cookiecutter key
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "test_template"
        
        # Create test data
        test_context = {"cookiecutter": {"project_name": "Test Project", "version": "1.0"}}
        replay_file = replay_dir / f"{template_name}.json"
        
        # Write test file
        with open(replay_file, 'w', encoding='utf-8') as f:
            json.dump(test_context, f, indent=2)
        
        # Load and verify
        result = load(replay_dir, template_name)
        assert result == test_context
        assert "cookiecutter" in result
    
    # Test 2: Load with template name already containing .json extension
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "test_template.json"
        
        # Create test data
        test_context = {"cookiecutter": {"key": "value"}}
        replay_file = replay_dir / template_name
        
        # Write test file
        with open(replay_file, 'w', encoding='utf-8') as f:
            json.dump(test_context, f, indent=2)
        
        # Load and verify
        result = load(replay_dir, template_name)
        assert result == test_context
    
    # Test 3: Load with string path instead of Path object
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = "another_template"
        
        # Create test data
        test_context = {"cookiecutter": {"name": "Test"}}
        replay_file = Path(tmpdir) / f"{template_name}.json"
        
        # Write test file
        with open(replay_file, 'w', encoding='utf-8') as f:
            json.dump(test_context, f, indent=2)
        
        # Load using string path
        result = load(tmpdir, template_name)
        assert result == test_context
    
    # Test 4: Error when context doesn't contain cookiecutter key
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "bad_template"
        
        # Create invalid test data (missing cookiecutter key)
        invalid_context = {"project_name": "Test Project"}
        replay_file = replay_dir / f"{template_name}.json"
        
        # Write test file
        with open(replay_file, 'w', encoding='utf-8') as f:
            json.dump(invalid_context, f, indent=2)
        
        # Should raise ValueError
        try:
            load(replay_dir, template_name)
            assert False, "Expected ValueError for missing cookiecutter key"
        except ValueError as e:
            assert "cookiecutter" in str(e)
    
    # Test 5: File not found error (implicitly tested via FileNotFoundError)
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "non_existent"
        
        # Should raise FileNotFoundError when trying to open the file
        try:
            load(replay_dir, template_name)
            assert False, "Expected FileNotFoundError"
        except FileNotFoundError:
            pass  # Expected behavior


# LLM-generated content at query #20
#--------------------------

```python
def test_dump():
    import tempfile
    import json
    import os
    
    # Test 1: Normal dump with valid context
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = "my_template"
        context = {"cookiecutter": {"project_name": "test", "version": "1.0"}}
        
        dump(tmpdir, template_name, context)
        
        expected_file = os.path.join(tmpdir, f"{template_name}.json")
        assert os.path.exists(expected_file)
        
        with open(expected_file, 'r', encoding='utf-8') as f:
            loaded_context = json.load(f)
        
        assert loaded_context == context
    
    # Test 2: Dump with template name already having .json extension
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = "my_template.json"
        context = {"cookiecutter": {"key": "value"}}
        
        dump(tmpdir, template_name, context)
        
        expected_file = os.path.join(tmpdir, template_name)
        assert os.path.exists(expected_file)
        
        with open(expected_file, 'r', encoding='utf-8') as f:
            loaded_context = json.load(f)
        
        assert loaded_context == context
    
    # Test 3: Dump raises ValueError when context lacks 'cookiecutter' key
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = "template"
        context = {"not_cookiecutter": {"key": "value"}}
        
        try:
            dump(tmpdir, template_name, context)
            assert False, "Expected ValueError"
        except ValueError as e:
            assert "Context is required to contain a cookiecutter key" in str(e)
    
    # Test 4: Dump creates directory if it doesn't exist
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = os.path.join(tmpdir, "nonexistent", "subdir")
        template_name = "template"
        context = {"cookiecutter": {"test": "data"}}
        
        assert not os.path.exists(replay_dir)
        dump(replay_dir, template_name, context)
        
        expected_file = os.path.join(replay_dir, f"{template_name}.json")
        assert os.path.exists(expected_file)
    
    # Test 5: Dump with nested context structure
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = "complex_template"
        context = {
            "cookiecutter": {
                "project": {
                    "name": "test",
                    "author": "developer",
                    "dependencies": ["dep1", "dep2"]
                },
                "settings": {"debug": True, "port": 8080}
            }
        }
        
        dump(tmpdir, template_name, context)
        
        expected_file = os.path.join(tmpdir, f"{template_name}.json")
        with open(expected_file, 'r', encoding='utf-8') as f:
            loaded_context = json.load(f)
        
        assert loaded_context == context


# LLM-generated content at query #21
#--------------------------

```python
def test_dump():
    import tempfile
    import json
    import os
    
    # Test 1: Normal dump with valid context
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = "my_template"
        context = {"cookiecutter": {"project_name": "Test Project", "version": "1.0"}}
        
        dump(tmpdir, template_name, context)
        
        expected_file = os.path.join(tmpdir, f"{template_name}.json")
        assert os.path.exists(expected_file)
        
        with open(expected_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        assert loaded_data == context
    
    # Test 2: Dump with existing .json extension in template_name
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = "my_template.json"
        context = {"cookiecutter": {"key": "value"}}
        
        dump(tmpdir, template_name, context)
        
        expected_file = os.path.join(tmpdir, template_name)
        assert os.path.exists(expected_file)
        
        with open(expected_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        assert loaded_data == context
    
    # Test 3: Dump with nested replay_dir path
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = os.path.join(tmpdir, "nested", "dir")
        template_name = "template"
        context = {"cookiecutter": {"test": "data"}}
        
        dump(replay_dir, template_name, context)
        
        expected_file = os.path.join(replay_dir, f"{template_name}.json")
        assert os.path.exists(expected_file)
        
        with open(expected_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        assert loaded_data == context
    
    # Test 4: Dump raises ValueError when context lacks 'cookiecutter' key
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = "test"
        context = {"not_cookiecutter": {"key": "value"}}
        
        try:
            dump(tmpdir, template_name, context)
            assert False, "Expected ValueError"
        except ValueError as e:
            assert "Context is required to contain a cookiecutter key" in str(e)
    
    # Test 5: Dump with complex context structure
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = "complex"
        context = {
            "cookiecutter": {
                "project_name": "My Project",
                "author": "John Doe",
                "features": ["feature1", "feature2"],
                "settings": {"debug": True, "port": 8080}
            }
        }
        
        dump(tmpdir, template_name, context)
        
        expected_file = os.path.join(tmpdir, f"{template_name}.json")
        assert os.path.exists(expected_file)
        
        with open(expected_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        assert loaded_data == context


# LLM-generated content at query #22
#--------------------------

```python
def test_load():
    import tempfile
    import json
    from pathlib import Path

    # Test normal case with valid cookiecutter context
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "test_template"
        
        # Create test data
        test_context = {"cookiecutter": {"project_name": "Test Project", "version": "1.0"}}
        replay_file = replay_dir / f"{template_name}.json"
        
        # Write test file
        replay_file.write_text(json.dumps(test_context))
        
        # Test load function
        result = load(replay_dir, template_name)
        assert result == test_context
        assert "cookiecutter" in result

    # Test with template_name that already has .json extension
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "test_template.json"
        
        # Create test data
        test_context = {"cookiecutter": {"key": "value"}}
        replay_file = replay_dir / template_name
        
        # Write test file
        replay_file.write_text(json.dumps(test_context))
        
        # Test load function
        result = load(replay_dir, template_name)
        assert result == test_context

    # Test with string path instead of Path object
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = "another_template"
        
        # Create test data
        test_context = {"cookiecutter": {"data": "test"}}
        replay_file = Path(tmpdir) / f"{template_name}.json"
        
        # Write test file
        replay_file.write_text(json.dumps(test_context))
        
        # Test load function with string path
        result = load(tmpdir, template_name)
        assert result == test_context

    # Test error case: missing cookiecutter key
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "invalid_template"
        
        # Create invalid test data (missing cookiecutter key)
        invalid_context = {"project_name": "Test Project"}
        replay_file = replay_dir / f"{template_name}.json"
        
        # Write test file
        replay_file.write_text(json.dumps(invalid_context))
        
        # Test that ValueError is raised
        try:
            load(replay_dir, template_name)
            assert False, "Expected ValueError for missing cookiecutter key"
        except ValueError as e:
            assert "Context is required to contain a cookiecutter key" in str(e)

    # Test error case: file not found
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "non_existent_template"
        
        # Test that FileNotFoundError is raised
        try:
            load(replay_dir, template_name)
            assert False, "Expected FileNotFoundError"
        except FileNotFoundError:
            pass  # Expected


# LLM-generated content at query #23
#--------------------------

```python
def test_dump():
    import tempfile
    import json
    import os
    
    # Test 1: Normal dump with valid context
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = "my_template"
        context = {"cookiecutter": {"project_name": "Test Project", "version": "1.0"}}
        
        dump(tmpdir, template_name, context)
        
        expected_file = os.path.join(tmpdir, f"{template_name}.json")
        assert os.path.exists(expected_file)
        
        with open(expected_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        assert loaded_data == context
    
    # Test 2: Dump with existing .json extension in template_name
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = "my_template.json"
        context = {"cookiecutter": {"key": "value"}}
        
        dump(tmpdir, template_name, context)
        
        expected_file = os.path.join(tmpdir, template_name)
        assert os.path.exists(expected_file)
        
        with open(expected_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        assert loaded_data == context
    
    # Test 3: Dump should raise ValueError when cookiecutter key is missing
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = "test_template"
        context = {"not_cookiecutter": {"key": "value"}}
        
        try:
            dump(tmpdir, template_name, context)
            assert False, "Expected ValueError to be raised"
        except ValueError as e:
            assert "Context is required to contain a cookiecutter key" in str(e)
    
    # Test 4: Dump creates directory if it doesn't exist
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = os.path.join(tmpdir, "nonexistent", "subdir")
        template_name = "template"
        context = {"cookiecutter": {"test": "data"}}
        
        assert not os.path.exists(replay_dir)
        dump(replay_dir, template_name, context)
        
        expected_file = os.path.join(replay_dir, f"{template_name}.json")
        assert os.path.exists(expected_file)
    
    # Test 5: Dump with nested context structure
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = "complex_template"
        context = {
            "cookiecutter": {
                "project": {
                    "name": "My Project",
                    "author": "Test Author",
                    "dependencies": ["dep1", "dep2"]
                },
                "settings": {
                    "debug": True,
                    "port": 8080
                }
            }
        }
        
        dump(tmpdir, template_name, context)
        
        expected_file = os.path.join(tmpdir, f"{template_name}.json")
        with open(expected_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        assert loaded_data == context


# LLM-generated content at query #24
#--------------------------

```python
def test_dump(tmp_path):
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {"cookiecutter": {"project_name": "Test Project", "version": "1.0"}}

    dump(replay_dir, template_name, context)

    expected_file = replay_dir / f"{template_name}.json"
    assert expected_file.exists()

    with open(expected_file, encoding="utf-8") as f:
        loaded_data = json.load(f)
    assert loaded_data == context


def test_dump_with_json_extension(tmp_path):
    replay_dir = tmp_path / "replay"
    template_name = "my_template.json"
    context = {"cookiecutter": {"key": "value"}}

    dump(replay_dir, template_name, context)

    expected_file = replay_dir / template_name
    assert expected_file.exists()

    with open(expected_file, encoding="utf-8") as f:
        loaded_data = json.load(f)
    assert loaded_data == context


def test_dump_missing_cookiecutter_key(tmp_path):
    replay_dir = tmp_path / "replay"
    template_name = "template"
    context = {"not_cookiecutter": {"key": "value"}}

    try:
        dump(replay_dir, template_name, context)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "Context is required to contain a cookiecutter key"


def test_dump_creates_directory(tmp_path):
    replay_dir = tmp_path / "new" / "nested" / "replay"
    template_name = "template"
    context = {"cookiecutter": {"test": "data"}}

    assert not replay_dir.exists()
    dump(replay_dir, template_name, context)
    assert replay_dir.exists()


def test_dump_valid_json_format(tmp_path):
    replay_dir = tmp_path / "replay"
    template_name = "test"
    context = {"cookiecutter": {"nested": {"key": "value"}, "list": [1, 2, 3]}}

    dump(replay_dir, template_name, context)

    output_file = replay_dir / f"{template_name}.json"
    with open(output_file, encoding="utf-8") as f:
        content = f.read()
        # Check for pretty print formatting (indent=2)
        assert '  ' in content
        # Verify it's valid JSON
        loaded = json.loads(content)
        assert loaded == context


# LLM-generated content at query #25
#--------------------------

```python
def test_dump():
    import tempfile
    import json
    import os
    
    # Test normal case with valid context
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = "my_template"
        context = {"cookiecutter": {"project_name": "Test Project", "version": "1.0"}}
        
        dump(tmpdir, template_name, context)
        
        expected_file = os.path.join(tmpdir, f"{template_name}.json")
        assert os.path.exists(expected_file)
        
        with open(expected_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        assert loaded_data == context
    
    # Test with template_name already having .json extension
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = "my_template.json"
        context = {"cookiecutter": {"key": "value"}}
        
        dump(tmpdir, template_name, context)
        
        expected_file = os.path.join(tmpdir, template_name)
        assert os.path.exists(expected_file)
        
        with open(expected_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        assert loaded_data == context
    
    # Test with Path object as replay_dir
    with tempfile.TemporaryDirectory() as tmpdir:
        from pathlib import Path
        path_obj = Path(tmpdir)
        template_name = "test_template"
        context = {"cookiecutter": {"data": "test"}}
        
        dump(path_obj, template_name, context)
        
        expected_file = os.path.join(tmpdir, f"{template_name}.json")
        assert os.path.exists(expected_file)
    
    # Test that missing cookiecutter key raises ValueError
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = "bad_template"
        context = {"not_cookiecutter": {"key": "value"}}
        
        try:
            dump(tmpdir, template_name, context)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "cookiecutter" in str(e)
    
    # Test that directory is created if it doesn't exist
    with tempfile.TemporaryDirectory() as tmpdir:
        nested_dir = os.path.join(tmpdir, "nested", "deep", "directory")
        template_name = "nested_template"
        context = {"cookiecutter": {"test": "data"}}
        
        assert not os.path.exists(nested_dir)
        dump(nested_dir, template_name, context)
        
        expected_file = os.path.join(nested_dir, f"{template_name}.json")
        assert os.path.exists(expected_file)


# LLM-generated content at query #26
#--------------------------

```python
def test_dump():
    import tempfile
    import json
    import os
    
    # Test 1: Normal dump with valid context
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = "my_template"
        context = {"cookiecutter": {"project_name": "Test Project", "version": "1.0"}}
        
        dump(tmpdir, template_name, context)
        
        expected_file = os.path.join(tmpdir, f"{template_name}.json")
        assert os.path.exists(expected_file)
        
        with open(expected_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        assert loaded_data == context
    
    # Test 2: Dump with template name already having .json extension
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = "my_template.json"
        context = {"cookiecutter": {"key": "value"}}
        
        dump(tmpdir, template_name, context)
        
        expected_file = os.path.join(tmpdir, template_name)
        assert os.path.exists(expected_file)
        
        with open(expected_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        assert loaded_data == context
    
    # Test 3: Dump creates directory if it doesn't exist
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = os.path.join(tmpdir, "subdir", "nested")
        template_name = "test"
        context = {"cookiecutter": {"data": "test"}}
        
        assert not os.path.exists(replay_dir)
        dump(replay_dir, template_name, context)
        
        expected_file = os.path.join(replay_dir, f"{template_name}.json")
        assert os.path.exists(expected_file)
    
    # Test 4: Dump raises ValueError when context doesn't have cookiecutter key
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = "test"
        context = {"not_cookiecutter": {"key": "value"}}
        
        try:
            dump(tmpdir, template_name, context)
            assert False, "Expected ValueError to be raised"
        except ValueError as e:
            assert str(e) == "Context is required to contain a cookiecutter key"
    
    # Test 5: Dump with complex nested context
    with tempfile.TemporaryDirectory() as tmpdir:
        template_name = "complex"
        context = {
            "cookiecutter": {
                "project": {
                    "name": "Test",
                    "settings": {"debug": True, "port": 8080}
                },
                "author": "Test Author",
                "dependencies": ["dep1", "dep2"]
            }
        }
        
        dump(tmpdir, template_name, context)
        
        expected_file = os.path.join(tmpdir, f"{template_name}.json")
        with open(expected_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        assert loaded_data == context


