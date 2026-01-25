####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

# Unit test for function get_file_name
def test_get_file_name():
    """Test the get_file_name function."""
    # Test with a template name that does not end with .json
    assert get_file_name('/tmp/replay', 'template') == '/tmp/replay/template.json'
    # Test with a template name that ends with .json
    assert get_file_name('/tmp/replay', 'template.json') == '/tmp/replay/template.json'
    # Test with a different replay directory
    assert get_file_name('/var/replay', 'template') == '/var/replay/template.json'


# LLM-generated content at query #2
#--------------------------

# Unit test for function get_file_name
def test_get_file_name():
    replay_dir = "/tmp/replay"
    template_name = "template"
    expected_file_name = "/tmp/replay/template.json"
    assert get_file_name(replay_dir, template_name) == expected_file_name

    template_name = "template.json"
    expected_file_name = "/tmp/replay/template.json"
    assert get_file_name(replay_dir, template_name) == expected_file_name



# LLM-generated content at query #3
#--------------------------

# Unit test for function get_file_name
def test_get_file_name():
    assert get_file_name("/path/to/replay", "template") == "/path/to/replay/template.json"
    assert get_file_name("/path/to/replay", "template.json") == "/path/to/replay/template.json"
    assert get_file_name("/path/to/replay", "template.xml") == "/path/to/replay/template.xml.json"



# LLM-generated content at query #4
#--------------------------

# Unit test for function get_file_name
def test_get_file_name():
    replay_dir = "test_dir"
    template_name = "test_template"
    expected_path = os.path.join(replay_dir, f"{template_name}.json")
    assert get_file_name(replay_dir, template_name) == expected_path



# LLM-generated content at query #5
#--------------------------

# Unit test for function get_file_name
def test_get_file_name():
    replay_dir = "/test/replay"
    template_name = "test_template"
    expected = "/test/replay/test_template.json"
    assert get_file_name(replay_dir, template_name) == expected

    template_name = "test_template.json"
    expected = "/test/replay/test_template.json"
    assert get_file_name(replay_dir, template_name) == expected


# LLM-generated content at query #6
#--------------------------

# Unit test for function load
def test_load():
    from pathlib import Path
    import tempfile
    import shutil

    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()

    try:
        # Define the replay directory and template name
        replay_dir = Path(temp_dir)
        template_name = "test_template"

        # Create a test context
        test_context = {"cookiecutter": {"key": "value"}}

        # Dump the context to a file
        dump(replay_dir, template_name, test_context)

        # Load the context from the file
        loaded_context = load(replay_dir, template_name)

        # Assert that the loaded context matches the original context
        assert loaded_context == test_context

    finally:
        # Clean up the temporary directory
        shutil.rmtree(temp_dir)


# LLM-generated content at query #7
#--------------------------

# Unit test for function dump
def test_dump():
    replay_dir = 'test_replay_dir'
    template_name = 'test_template'
    context = {'cookiecutter': {'key': 'value'}}
    
    try:
        dump(replay_dir, template_name, context)
        with open(get_file_name(replay_dir, template_name), 'r', encoding="utf-8") as file:
            loaded_context = json.load(file)
        assert loaded_context == context
    finally:
        os.remove(get_file_name(replay_dir, template_name))
        os.rmdir(replay_dir)



# LLM-generated content at query #8
#--------------------------

# Unit test for function load
def test_load():
    """Test the load function."""
    import tempfile
    from pathlib import Path

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        replay_dir = Path(tmp_dir)
        template_name = "test_template"
        test_context = {"cookiecutter": {"key": "value"}}

        # Save test context to a file
        dump(replay_dir, template_name, test_context)

        # Load the context from the file
        loaded_context = load(replay_dir, template_name)

        # Verify the loaded context matches the original
        assert loaded_context == test_context

        # Test with non-existent file (should raise FileNotFoundError)
        try:
            load(replay_dir, "nonexistent_template")
            assert False, "Expected FileNotFoundError"
        except FileNotFoundError:
            pass

        # Test with invalid JSON (should raise json.JSONDecodeError)
        invalid_json_file = get_file_name(replay_dir, "invalid_json")
        with open(invalid_json_file, "w", encoding="utf-8") as f:
            f.write("invalid json")
        try:
            load(replay_dir, "invalid_json")
            assert False, "Expected json.JSONDecodeError"
        except json.JSONDecodeError:
            pass

        # Test with missing cookiecutter key (should raise ValueError)
        invalid_context_file = get_file_name(replay_dir, "invalid_context")
        with open(invalid_context_file, "w", encoding="utf-8") as f:
            json.dump({"key": "value"}, f)
        try:
            load(replay_dir, "invalid_context")
            assert False, "Expected ValueError"
        except ValueError as e:
            assert str(e) == "Context is required to contain a cookiecutter key"


# LLM-generated content at query #9
#--------------------------

# Unit test for function load
def test_load():
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}

    # Ensure the directory exists
    os.makedirs(replay_dir, exist_ok=True)

    # Create a test replay file
    replay_file = get_file_name(replay_dir, template_name)
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Load the context from the replay file
    loaded_context = load(replay_dir, template_name)

    # Assert that the loaded context matches the original context
    assert loaded_context == context

    # Clean up the test replay file
    os.remove(replay_file)
    os.rmdir(replay_dir)



# LLM-generated content at query #10
#--------------------------

# Unit test for function dump
def test_dump():
    import tempfile
    import shutil
    import json

    # Create a temporary directory
    replay_dir = tempfile.mkdtemp()

    try:
        template_name = "test_template"
        context = {"cookiecutter": {"key": "value"}}

        # Call the dump function
        dump(replay_dir, template_name, context)

        # Verify the file was created
        file_path = get_file_name(replay_dir, template_name)
        assert os.path.exists(file_path)

        # Verify the content of the file
        with open(file_path, "r", encoding="utf-8") as file:
            loaded_context = json.load(file)
        assert loaded_context == context

    finally:
        # Clean up the temporary directory
        shutil.rmtree(replay_dir)



# LLM-generated content at query #11
#--------------------------

# Unit test for function dump
def test_dump():
    import tempfile
    import shutil

    replay_dir = tempfile.mkdtemp()
    template_name = "test_template"
    context = {"cookiecutter": {"test_key": "test_value"}}

    dump(replay_dir, template_name, context)

    file_path = os.path.join(replay_dir, f"{template_name}.json")
    assert os.path.exists(file_path)

    with open(file_path, 'r', encoding="utf-8") as infile:
        loaded_context = json.load(infile)
        assert loaded_context == context

    shutil.rmtree(replay_dir)



# LLM-generated content at query #12
#--------------------------

# Unit test for function load
def test_load():
    """Test the load function."""
    import tempfile
    from pathlib import Path

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        replay_dir = Path(tmp_dir)
        template_name = "test_template"
        
        # Create a test context
        test_context = {"cookiecutter": {"key": "value"}}
        
        # Save the test context to a file
        dump(replay_dir, template_name, test_context)
        
        # Load the context back
        loaded_context = load(replay_dir, template_name)
        
        # Verify the loaded context matches the original
        assert loaded_context == test_context


# LLM-generated content at query #13
#--------------------------

# Unit test for function load
def test_load():
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    
    # Ensure the directory exists
    if not os.path.exists(replay_dir):
        os.makedirs(replay_dir)
    
    # Dump the context to a file
    dump(replay_dir, template_name, context)
    
    # Load the context from the file
    loaded_context = load(replay_dir, template_name)
    
    # Assert that the loaded context matches the original context
    assert loaded_context == context
    
    # Clean up: remove the test file and directory
    os.remove(get_file_name(replay_dir, template_name))
    os.rmdir(replay_dir)


# LLM-generated content at query #14
#--------------------------

# Unit test for function dump
def test_dump():
    import tempfile
    import shutil

    replay_dir = tempfile.mkdtemp()
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "Test Project"}}

    dump(replay_dir, template_name, context)

    replay_file = get_file_name(replay_dir, template_name)
    assert os.path.exists(replay_file)

    with open(replay_file, "r", encoding="utf-8") as infile:
        loaded_context = json.load(infile)
        assert loaded_context == context

    shutil.rmtree(replay_dir)

    try:
        dump(replay_dir, template_name, {"invalid_key": "value"})
    except ValueError as e:
        assert str(e) == "Context is required to contain a cookiecutter key"



# LLM-generated content at query #15
#--------------------------

# Unit test for function load
def test_load():
    import tempfile
    from pathlib import Path

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdirname:
        replay_dir = Path(tmpdirname)
        template_name = "test_template"

        # Create a context dictionary
        context = {"cookiecutter": {"key1": "value1", "key2": "value2"}}

        # Write the context to a file
        dump(replay_dir, template_name, context)

        # Load the context from the file
        loaded_context = load(replay_dir, template_name)

        # Assert that the loaded context matches the original context
        assert loaded_context == context

        # Test loading with a non-existing file
        try:
            load(replay_dir, "non_existing_template")
            assert False, "Expected ValueError when loading a non-existing file"
        except ValueError:
            pass

        # Test loading with a file that does not contain 'cookiecutter' key
        invalid_context = {"key1": "value1", "key2": "value2"}
        invalid_file = get_file_name(replay_dir, "invalid_template")
        with open(invalid_file, 'w', encoding="utf-8") as outfile:
            json.dump(invalid_context, outfile, indent=2)

        try:
            load(replay_dir, "invalid_template")
            assert False, "Expected ValueError when file does not contain 'cookiecutter' key"
        except ValueError:
            pass


# LLM-generated content at query #16
#--------------------------

# Unit test for function load
def test_load():
    # Arrange
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    expected_context = {"cookiecutter": {"key": "value"}}
    
    # Create a test JSON file
    os.makedirs(replay_dir, exist_ok=True)
    test_file_path = os.path.join(replay_dir, template_name + ".json")
    with open(test_file_path, "w", encoding="utf-8") as file:
        json.dump(expected_context, file)
    
    # Act
    loaded_context = load(replay_dir, template_name)
    
    # Assert
    assert loaded_context == expected_context
    
    # Cleanup
    os.remove(test_file_path)
    os.rmdir(replay_dir)



# LLM-generated content at query #17
#--------------------------

# Unit test for function load
def test_load():
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}

    # Dump context to file
    dump(replay_dir, template_name, context)

    # Load context from file
    loaded_context = load(replay_dir, template_name)

    # Assert the loaded context is equal to the original context
    assert loaded_context == context

    # Clean up
    os.remove(get_file_name(replay_dir, template_name))
    os.rmdir(replay_dir)


# LLM-generated content at query #18
#--------------------------

# Unit test for function load
def test_load():
    import tempfile
    from pathlib import Path

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdirname:
        replay_dir = Path(tmpdirname)
        template_name = "test_template"
        context = {"cookiecutter": {"key": "value"}}

        # Write the context to a file
        dump(replay_dir, template_name, context)

        # Load the context from the file
        loaded_context = load(replay_dir, template_name)

        # Assert that the loaded context matches the original context
        assert loaded_context == context



# LLM-generated content at query #19
#--------------------------

# Unit test for function dump
def test_dump():
    import tempfile
    import shutil
    from pathlib import Path

    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    replay_dir = Path(temp_dir)
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}

    # Call the dump function
    dump(replay_dir, template_name, context)

    # Verify the file was created
    file_path = replay_dir / f"{template_name}.json"
    assert file_path.exists()

    # Verify the content of the file
    with open(file_path, 'r', encoding='utf-8') as infile:
        loaded_context = json.load(infile)
        assert loaded_context == context

    # Clean up
    shutil.rmtree(temp_dir)



# LLM-generated content at query #20
#--------------------------

# Unit test for function dump
def test_dump():
    import tempfile
    import shutil
    
    replay_dir = tempfile.mkdtemp()
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    
    dump(replay_dir, template_name, context)
    
    replay_file = get_file_name(replay_dir, template_name)
    assert os.path.exists(replay_file)
    
    with open(replay_file, 'r') as infile:
        loaded_context = json.load(infile)
    
    assert loaded_context == context
    
    shutil.rmtree(replay_dir)



# LLM-generated content at query #21
#--------------------------

# Unit test for function load
def test_load():
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    test_context = {"cookiecutter": {"key": "value"}}
    dump(replay_dir, template_name, test_context)
    loaded_context = load(replay_dir, template_name)
    assert loaded_context == test_context



# LLM-generated content at query #22
#--------------------------

# Unit test for function load
def test_load():
    """Test the load function."""
    import tempfile
    from pathlib import Path

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "test_template"
        test_context = {"cookiecutter": {"key": "value"}}

        # Save test context to a file
        dump(replay_dir, template_name, test_context)

        # Load the context back
        loaded_context = load(replay_dir, template_name)

        # Verify the loaded context matches the original
        assert loaded_context == test_context

        # Test with invalid context (no cookiecutter key)
        invalid_file = replay_dir / "invalid.json"
        with open(invalid_file, 'w', encoding="utf-8") as f:
            json.dump({"invalid": "data"}, f)

        try:
            load(replay_dir, "invalid")
            assert False, "Expected ValueError"
        except ValueError as e:
            assert str(e) == "Context is required to contain a cookiecutter key"


# LLM-generated content at query #23
#--------------------------

# Unit test for function load
def test_load():
    """Test the load function."""
    import tempfile
    from pathlib import Path

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        replay_dir = Path(tmp_dir)
        template_name = "test_template"
        test_context = {"cookiecutter": {"key": "value"}}

        # Write test context to file
        dump(replay_dir, template_name, test_context)

        # Load the context back
        loaded_context = load(replay_dir, template_name)

        # Verify the loaded context matches the original
        assert loaded_context == test_context

        # Test with non-existent template name
        try:
            load(replay_dir, "non_existent_template")
            assert False, "Expected ValueError for non-existent template"
        except (FileNotFoundError, ValueError):
            pass

        # Test with invalid context (missing cookiecutter key)
        invalid_file = replay_dir / "invalid_template.json"
        with open(invalid_file, 'w', encoding="utf-8") as f:
            json.dump({"invalid_key": "value"}, f)

        try:
            load(replay_dir, "invalid_template")
            assert False, "Expected ValueError for missing cookiecutter key"
        except ValueError:
            pass


# LLM-generated content at query #24
#--------------------------

# Unit test for function dump
def test_dump():
    replay_dir = "/tmp/replay"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    dump(replay_dir, template_name, context)
    assert os.path.exists(os.path.join(replay_dir, f"{template_name}.json"))



# LLM-generated content at query #25
#--------------------------

# Unit test for function load
def test_load():
    import tempfile
    from pathlib import Path

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdirname:
        replay_dir = Path(tmpdirname)
        template_name = "test_template"
        context = {"cookiecutter": {"key": "value"}}

        # Write context to file
        dump(replay_dir, template_name, context)

        # Load context from file
        loaded_context = load(replay_dir, template_name)

        # Assert that the loaded context is the same as the original context
        assert loaded_context == context



# LLM-generated content at query #26
#--------------------------

# Unit test for function load
def test_load():
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}

    # Ensure directory exists
    make_sure_path_exists(replay_dir)

    # Write test context to file
    replay_file = get_file_name(replay_dir, template_name)
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Load context from file
    loaded_context = load(replay_dir, template_name)

    # Verify loaded context matches original context
    assert loaded_context == context

    # Clean up test file
    os.remove(replay_file)



# LLM-generated content at query #27
#--------------------------

# Unit test for function dump
def test_dump():
    import tempfile
    from pathlib import Path

    replay_dir = tempfile.mkdtemp()
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}

    dump(replay_dir, template_name, context)

    replay_file = Path(replay_dir) / f"{template_name}.json"
    assert replay_file.exists()

    with open(replay_file, encoding="utf-8") as infile:
        loaded_context = json.load(infile)

    assert loaded_context == context



# LLM-generated content at query #28
#--------------------------

# Unit test for function dump
def test_dump():
    import tempfile
    import shutil

    replay_dir = tempfile.mkdtemp()
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}

    dump(replay_dir, template_name, context)

    replay_file = get_file_name(replay_dir, template_name)
    assert os.path.exists(replay_file)

    with open(replay_file, "r", encoding="utf-8") as f:
        loaded_context = json.load(f)
        assert loaded_context == context

    shutil.rmtree(replay_dir)



# LLM-generated content at query #29
#--------------------------

# Unit test for function dump
def test_dump():
    import tempfile
    import shutil
    from pathlib import Path

    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    replay_dir = Path(temp_dir)
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}

    try:
        # Test dumping the context
        dump(replay_dir, template_name, context)
        
        # Verify the file was created
        replay_file = replay_dir / f"{template_name}.json"
        assert replay_file.exists()
        
        # Verify the content
        with open(replay_file, encoding="utf-8") as f:
            loaded_context = json.load(f)
        assert loaded_context == context

    finally:
        # Clean up
        shutil.rmtree(temp_dir)


# LLM-generated content at query #30
#--------------------------

# Unit test for function load
def test_load():
    """Test the load function."""
    import tempfile
    from pathlib import Path

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "test_template"
        test_context = {"cookiecutter": {"key": "value"}}

        # Save test context to a file
        dump(replay_dir, template_name, test_context)

        # Load the context back
        loaded_context = load(replay_dir, template_name)

        # Verify the loaded context matches the original
        assert loaded_context == test_context

        # Test with non-existent template (should raise FileNotFoundError)
        try:
            load(replay_dir, "non_existent_template")
        except FileNotFoundError:
            pass
        else:
            assert False, "Expected FileNotFoundError"

        # Test with invalid context (missing cookiecutter key)
        invalid_file = replay_dir / "invalid_template.json"
        with open(invalid_file, "w", encoding="utf-8") as f:
            json.dump({"invalid": "context"}, f)

        try:
            load(replay_dir, "invalid_template")
        except ValueError as e:
            assert str(e) == "Context is required to contain a cookiecutter key"
        else:
            assert False, "Expected ValueError"


# LLM-generated content at query #31
#--------------------------

# Unit test for function load
def test_load():
    import tempfile
    from pathlib import Path

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdirname:
        replay_dir = Path(tmpdirname)
        template_name = 'test_template'
        context = {'cookiecutter': {'key': 'value'}}

        # Write context to a file
        dump(replay_dir, template_name, context)

        # Load context from the file
        loaded_context = load(replay_dir, template_name)

        # Assert that the loaded context matches the original context
        assert loaded_context == context



# LLM-generated content at query #32
#--------------------------

# Unit test for function load
def test_load():
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    
    # Ensure the directory exists
    make_sure_path_exists(replay_dir)
    
    # Dump the context to a file
    dump(replay_dir, template_name, context)
    
    # Load the context from the file
    loaded_context = load(replay_dir, template_name)
    
    # Assert the loaded context matches the original context
    assert loaded_context == context
    
    # Clean up: remove the test file
    os.remove(get_file_name(replay_dir, template_name))


# LLM-generated content at query #33
#--------------------------

# Unit test for function load
def test_load():
    """Test the load function."""
    import tempfile
    from pathlib import Path

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        replay_dir = Path(tmp_dir)
        template_name = "test_template"
        test_context = {"cookiecutter": {"key": "value"}}

        # Write test context to file
        dump(replay_dir, template_name, test_context)

        # Load the context back
        loaded_context = load(replay_dir, template_name)

        # Verify the loaded context matches the original
        assert loaded_context == test_context

        # Test with non-existent file
        try:
            load(replay_dir, "non_existent_template")
        except FileNotFoundError:
            pass
        else:
            assert False, "Expected FileNotFoundError"

        # Test with invalid context (missing cookiecutter key)
        invalid_file = replay_dir / "invalid_template.json"
        with open(invalid_file, "w", encoding="utf-8") as f:
            json.dump({"key": "value"}, f)

        try:
            load(replay_dir, "invalid_template")
        except ValueError as e:
            assert str(e) == "Context is required to contain a cookiecutter key"
        else:
            assert False, "Expected ValueError"


# LLM-generated content at query #34
#--------------------------

# Unit test for function load
def test_load():
    """Test the load function."""
    import tempfile
    from pathlib import Path

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "test_template"
        test_context = {"cookiecutter": {"key": "value"}}

        # Save test context to a file
        dump(replay_dir, template_name, test_context)

        # Load the context back
        loaded_context = load(replay_dir, template_name)

        # Verify the loaded context matches the original
        assert loaded_context == test_context

        # Test with invalid context (no 'cookiecutter' key)
        invalid_file = replay_dir / "invalid.json"
        with open(invalid_file, "w", encoding="utf-8") as f:
            json.dump({"key": "value"}, f)

        try:
            load(replay_dir, "invalid")
            assert False, "Expected ValueError"
        except ValueError:
            pass


# LLM-generated content at query #35
#--------------------------

# Unit test for function dump
def test_dump():
    import tempfile
    import shutil

    replay_dir = tempfile.mkdtemp()
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "Test Project"}}

    try:
        dump(replay_dir, template_name, context)
        assert os.path.exists(os.path.join(replay_dir, f"{template_name}.json"))
    finally:
        shutil.rmtree(replay_dir)



# LLM-generated content at query #36
#--------------------------

# Unit test for function dump
def test_dump():
    import tempfile
    import shutil
    from pathlib import Path

    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    replay_dir = Path(temp_dir) / "replay"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}

    try:
        # Test dump function
        dump(replay_dir, template_name, context)
        
        # Verify the file was created
        expected_file = replay_dir / f"{template_name}.json"
        assert expected_file.exists()

        # Verify the content
        with open(expected_file, "r", encoding="utf-8") as f:
            loaded_content = json.load(f)
        assert loaded_content == context

    finally:
        # Clean up
        shutil.rmtree(temp_dir)


# LLM-generated content at query #37
#--------------------------

# Unit test for function load
def test_load():
    """Test the load function."""
    import tempfile
    from pathlib import Path

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        replay_dir = Path(tmp_dir)
        template_name = "test_template"
        test_context = {"cookiecutter": {"key": "value"}}

        # Write test context to file
        dump(replay_dir, template_name, test_context)

        # Load the context back
        loaded_context = load(replay_dir, template_name)

        # Verify the loaded context matches the original
        assert loaded_context == test_context

        # Test with invalid context (no cookiecutter key)
        invalid_file = replay_dir / "invalid.json"
        with open(invalid_file, 'w', encoding="utf-8") as f:
            json.dump({"invalid": "data"}, f)

        try:
            load(replay_dir, "invalid")
            assert False, "Expected ValueError"
        except ValueError:
            pass


# LLM-generated content at query #38
#--------------------------

# Unit test for function load
def test_load():
    """Test the load function."""
    import tempfile
    from pathlib import Path

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "test_template"
        test_context = {"cookiecutter": {"key": "value"}}

        # Write test context to file
        test_file = replay_dir / f"{template_name}.json"
        with open(test_file, 'w', encoding="utf-8") as f:
            json.dump(test_context, f)

        # Test load function
        loaded_context = load(replay_dir, template_name)
        assert loaded_context == test_context

        # Test with invalid context (no cookiecutter key)
        invalid_context = {"key": "value"}
        with open(test_file, 'w', encoding="utf-8") as f:
            json.dump(invalid_context, f)

        try:
            load(replay_dir, template_name)
            assert False, "Expected ValueError"
        except ValueError:
            pass


# LLM-generated content at query #39
#--------------------------

# Unit test for function dump
def test_dump():
    """Test the dump function."""
    import tempfile
    import shutil

    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    try:
        # Test data
        template_name = "test_template"
        context = {"cookiecutter": {"key": "value"}}

        # Call the function
        dump(temp_dir, template_name, context)

        # Verify the file was created
        expected_file = os.path.join(temp_dir, f"{template_name}.json")
        assert os.path.exists(expected_file)

        # Verify the content
        with open(expected_file, 'r', encoding='utf-8') as f:
            content = json.load(f)
        assert content == context

    finally:
        # Clean up
        shutil.rmtree(temp_dir)


# LLM-generated content at query #40
#--------------------------

# Unit test for function dump
def test_dump():
    replay_dir = "test_replay"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}

    dump(replay_dir, template_name, context)

    replay_file = get_file_name(replay_dir, template_name)
    assert os.path.exists(replay_file)

    with open(replay_file, "r", encoding="utf-8") as f:
        loaded_context = json.load(f)
        assert loaded_context == context

    os.remove(replay_file)
    os.rmdir(replay_dir)



# LLM-generated content at query #41
#--------------------------

# Unit test for function dump
def test_dump():
    import tempfile
    from pathlib import Path

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdirname:
        replay_dir = Path(tmpdirname)
        template_name = "test_template"
        context = {"cookiecutter": {"key": "value"}}

        # Call dump function
        dump(replay_dir, template_name, context)

        # Verify file exists
        file_path = replay_dir / f"{template_name}.json"
        assert file_path.exists()

        # Verify file content
        with open(file_path, "r", encoding="utf-8") as f:
            file_content = json.load(f)
        assert file_content == context



# LLM-generated content at query #42
#--------------------------

# Unit test for function load
def test_load():
    import tempfile
    from pathlib import Path

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdirname:
        replay_dir = Path(tmpdirname)
        template_name = "test_template"
        context = {"cookiecutter": {"key": "value"}}

        # Write context to file
        dump(replay_dir, template_name, context)

        # Load context from file
        loaded_context = load(replay_dir, template_name)

        # Assert that the loaded context matches the original context
        assert loaded_context == context, "Loaded context does not match the original context"

        # Test with invalid context (missing 'cookiecutter' key)
        invalid_context = {"key": "value"}
        invalid_file = get_file_name(replay_dir, "invalid_template")
        with open(invalid_file, 'w', encoding="utf-8") as outfile:
            json.dump(invalid_context, outfile, indent=2)

        try:
            load(replay_dir, "invalid_template")
            assert False, "Expected ValueError when loading invalid context"
        except ValueError:
            pass


# LLM-generated content at query #43
#--------------------------

# Unit test for function dump
def test_dump():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as temp_dir:
        replay_dir = Path(temp_dir)
        template_name = "test_template"
        context = {"cookiecutter": {"key": "value"}}

        dump(replay_dir, template_name, context)

        replay_file = replay_dir / f"{template_name}.json"
        assert replay_file.exists()

        with open(replay_file, "r", encoding="utf-8") as f:
            loaded_context = json.load(f)
            assert loaded_context == context



# LLM-generated content at query #44
#--------------------------

# Unit test for function load
def test_load():
    import tempfile
    import shutil
    from pathlib import Path

    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()

    # Define the template name and context
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "Test Project"}}

    # Write the context to a JSON file
    replay_file = Path(temp_dir) / f"{template_name}.json"
    with open(replay_file, 'w', encoding="utf-8") as outfile:
        json.dump(context, outfile, indent=2)

    # Load the context from the JSON file
    loaded_context = load(temp_dir, template_name)

    # Assert that the loaded context matches the original context
    assert loaded_context == context

    # Clean up the temporary directory
    shutil.rmtree(temp_dir)


# LLM-generated content at query #45
#--------------------------

# Unit test for function load
def test_load():
    """Test the load function."""
    import tempfile
    from pathlib import Path

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "test_template"
        context = {"cookiecutter": {"key": "value"}}

        # Save test data
        dump(replay_dir, template_name, context)

        # Load the data and verify
        loaded_context = load(replay_dir, template_name)
        assert loaded_context == context

        # Test with non-existent template name
        try:
            load(replay_dir, "non_existent_template")
            assert False, "Expected ValueError for non-existent template"
        except (FileNotFoundError, ValueError):
            pass

        # Test with invalid context (no cookiecutter key)
        invalid_file = replay_dir / "invalid.json"
        with open(invalid_file, 'w', encoding="utf-8") as f:
            json.dump({"no_cookiecutter": True}, f)
        try:
            load(replay_dir, "invalid")
            assert False, "Expected ValueError for invalid context"
        except ValueError:
            pass


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

# Unit test for function load
def test_load():
    test_dir = "test_replay_dir"
    test_template = "test_template"
    test_context = {"cookiecutter": {"key": "value"}}
    
    # Ensure the directory exists
    os.makedirs(test_dir, exist_ok=True)
    
    # Save the context to a file
    dump(test_dir, test_template, test_context)
    
    # Load the context from the file
    loaded_context = load(test_dir, test_template)
    
    # Assert that the loaded context matches the original context
    assert loaded_context == test_context, "Loaded context does not match the original context"
    
    # Clean up the test directory
    os.remove(os.path.join(test_dir, f"{test_template}.json"))
    os.rmdir(test_dir)

# Run the unit test
test_load()


# LLM-generated content at query #2
#--------------------------

# Unit test for function load
def test_load():
    """Test the load function."""
    import tempfile
    from pathlib import Path

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "test_template"
        test_context = {"cookiecutter": {"key": "value"}}

        # Write test data to file
        dump(replay_dir, template_name, test_context)

        # Load the data back
        loaded_context = load(replay_dir, template_name)

        # Verify the loaded data matches the original
        assert loaded_context == test_context

        # Test with invalid context (no cookiecutter key)
        invalid_file = replay_dir / "invalid.json"
        with open(invalid_file, 'w', encoding="utf-8") as f:
            json.dump({"invalid": "data"}, f)

        try:
            load(replay_dir, "invalid")
            assert False, "Expected ValueError"
        except ValueError:
            pass


# LLM-generated content at query #3
#--------------------------

# Unit test for function dump
def test_dump():
    """Test the dump function."""
    import tempfile
    import shutil

    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    try:
        # Test data
        template_name = "test_template"
        context = {"cookiecutter": {"key": "value"}}

        # Call the function
        dump(temp_dir, template_name, context)

        # Verify the file was created
        file_path = os.path.join(temp_dir, f"{template_name}.json")
        assert os.path.exists(file_path)

        # Verify the content
        with open(file_path, encoding="utf-8") as f:
            loaded_context = json.load(f)
        assert loaded_context == context

    finally:
        # Clean up
        shutil.rmtree(temp_dir)


# LLM-generated content at query #4
#--------------------------

# Unit test for function dump
def test_dump():
    import tempfile
    import shutil
    from pathlib import Path

    replay_dir = tempfile.mkdtemp()
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}

    dump(replay_dir, template_name, context)

    replay_file = Path(replay_dir) / f"{template_name}.json"
    assert replay_file.exists()

    with open(replay_file, 'r') as file:
        loaded_context = json.load(file)
        assert loaded_context == context

    shutil.rmtree(replay_dir)



# LLM-generated content at query #5
#--------------------------

# Unit test for function dump
def test_dump():
    import tempfile
    import shutil
    import json
    from pathlib import Path

    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    replay_dir = Path(temp_dir)
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "Test Project"}}

    try:
        # Test dump function
        dump(replay_dir, template_name, context)
        
        # Verify file was created
        replay_file = replay_dir / f"{template_name}.json"
        assert replay_file.exists()

        # Verify content
        with open(replay_file, encoding="utf-8") as f:
            loaded_context = json.load(f)
        assert loaded_context == context

    finally:
        # Clean up
        shutil.rmtree(temp_dir)


# LLM-generated content at query #6
#--------------------------

# Unit test for function load
def test_load():
    """Test the load function."""
    import tempfile
    from pathlib import Path

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Prepare test data
        test_context = {'cookiecutter': {'key': 'value'}}
        template_name = 'test_template'
        
        # Write the test data to a file
        replay_dir = Path(tmpdir)
        dump(replay_dir, template_name, test_context)
        
        # Load the data back
        loaded_context = load(replay_dir, template_name)
        
        # Assert the loaded data matches the original
        assert loaded_context == test_context


# LLM-generated content at query #7
#--------------------------

# Unit test for function dump
def test_dump():
    """Test the dump function."""
    import tempfile
    import shutil
    from pathlib import Path

    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    replay_dir = Path(temp_dir)
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}

    try:
        # Test dumping the context
        dump(replay_dir, template_name, context)
        
        # Verify the file was created
        file_path = replay_dir / f"{template_name}.json"
        assert file_path.exists(), "File was not created"

        # Verify the content of the file
        with open(file_path, 'r', encoding="utf-8") as f:
            loaded_context = json.load(f)
        assert loaded_context == context, "Context does not match"

    finally:
        # Clean up
        shutil.rmtree(temp_dir)


# LLM-generated content at query #8
#--------------------------

# Unit test for function load
def test_load():
    """Test the load function."""
    import tempfile
    from pathlib import Path

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        replay_dir = Path(tmp_dir)
        template_name = "test_template"
        test_context = {"cookiecutter": {"key": "value"}}

        # Save test context to a file
        dump(replay_dir, template_name, test_context)

        # Load the context back
        loaded_context = load(replay_dir, template_name)

        # Verify the loaded context matches the original
        assert loaded_context == test_context

        # Test with non-existent template name (should raise FileNotFoundError)
        try:
            load(replay_dir, "nonexistent_template")
            assert False, "Expected FileNotFoundError"
        except FileNotFoundError:
            pass

        # Test with invalid context (missing cookiecutter key)
        invalid_file = replay_dir / "invalid_template.json"
        with open(invalid_file, "w", encoding="utf-8") as f:
            json.dump({"invalid": "data"}, f)

        try:
            load(replay_dir, "invalid_template")
            assert False, "Expected ValueError"
        except ValueError as e:
            assert str(e) == "Context is required to contain a cookiecutter key"


# LLM-generated content at query #9
#--------------------------

# Unit test for function load
def test_load():
    """Test the load function."""
    import tempfile
    from pathlib import Path

    replay_dir = tempfile.mkdtemp()
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}

    # Write context to a file
    dump(replay_dir, template_name, context)

    # Load context from the file
    loaded_context = load(replay_dir, template_name)

    assert loaded_context == context, "Loaded context does not match the original context"

    # Clean up
    os.remove(Path(replay_dir) / f"{template_name}.json")
    os.rmdir(replay_dir)


# LLM-generated content at query #10
#--------------------------

# Unit test for function dump
def test_dump():
    import tempfile
    import shutil
    from pathlib import Path

    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    replay_dir = Path(temp_dir)

    try:
        # Test data
        template_name = "test_template"
        context = {"cookiecutter": {"key": "value"}}

        # Call the function
        dump(replay_dir, template_name, context)

        # Verify the file was created
        expected_file = replay_dir / f"{template_name}.json"
        assert expected_file.exists()

        # Verify the content
        with open(expected_file, "r", encoding="utf-8") as f:
            content = json.load(f)
        assert content == context

    finally:
        # Clean up
        shutil.rmtree(temp_dir)


# LLM-generated content at query #11
#--------------------------

# Unit test for function dump
def test_dump():
    replay_dir = "test_replay"
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "Test Project",
            "author": "Test Author"
        }
    }
    dump(replay_dir, template_name, context)
    assert os.path.exists(os.path.join(replay_dir, f"{template_name}.json"))



# LLM-generated content at query #12
#--------------------------

# Unit test for function load
def test_load():
    """Test the load function."""
    import tempfile
    import shutil

    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    replay_dir = os.path.join(temp_dir, 'replay')
    os.makedirs(replay_dir)

    # Create a test JSON file
    test_data = {'cookiecutter': {'key': 'value'}}
    test_file = os.path.join(replay_dir, 'test.json')
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f)

    # Test loading the file
    loaded_data = load(replay_dir, 'test')
    assert loaded_data == test_data

    # Clean up
    shutil.rmtree(temp_dir)


# LLM-generated content at query #13
#--------------------------

# Unit test for function load
def test_load():
    import tempfile
    from pathlib import Path

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Setup test data
        template_name = "test_template"
        context = {"cookiecutter": {"key": "value"}}
        replay_dir = Path(temp_dir)
        replay_file = replay_dir / f"{template_name}.json"

        # Write test data to file
        with open(replay_file, "w", encoding="utf-8") as file:
            json.dump(context, file)

        # Test load function
        loaded_context = load(replay_dir, template_name)
        assert loaded_context == context

        # Test invalid context (missing 'cookiecutter' key)
        invalid_context = {"key": "value"}
        with open(replay_file, "w", encoding="utf-8") as file:
            json.dump(invalid_context, file)

        try:
            load(replay_dir, template_name)
        except ValueError as e:
            assert str(e) == "Context is required to contain a cookiecutter key"



# LLM-generated content at query #14
#--------------------------

# Unit test for function load
def test_load():
    """Test the load function."""
    import tempfile
    import shutil
    from pathlib import Path

    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    replay_dir = Path(temp_dir)
    template_name = "test_template"
    test_context = {"cookiecutter": {"key": "value"}}

    try:
        # Test file creation and loading
        dump(replay_dir, template_name, test_context)
        loaded_context = load(replay_dir, template_name)
        assert loaded_context == test_context

        # Test with invalid context (missing cookiecutter key)
        invalid_context = {"key": "value"}
        invalid_file = replay_dir / "invalid_template.json"
        with open(invalid_file, 'w', encoding="utf-8") as f:
            json.dump(invalid_context, f)
        try:
            load(replay_dir, "invalid_template")
            assert False, "Expected ValueError"
        except ValueError:
            pass

    finally:
        # Clean up
        shutil.rmtree(temp_dir)


# LLM-generated content at query #15
#--------------------------

# Unit test for function load
def test_load():
    """Test the load function."""
    import tempfile
    import shutil
    from pathlib import Path

    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    replay_dir = Path(temp_dir)
    template_name = "test_template"
    test_context = {"cookiecutter": {"key": "value"}}

    try:
        # Write test data
        dump(replay_dir, template_name, test_context)
        
        # Load and verify
        loaded_context = load(replay_dir, template_name)
        assert loaded_context == test_context
        
        # Test with non-existent file
        try:
            load(replay_dir, "nonexistent_template")
            assert False, "Expected ValueError"
        except ValueError:
            pass
            
        # Test with invalid context (missing cookiecutter key)
        invalid_file = replay_dir / "invalid.json"
        with open(invalid_file, 'w', encoding='utf-8') as f:
            json.dump({"invalid": "data"}, f)
        try:
            load(replay_dir, "invalid")
            assert False, "Expected ValueError"
        except ValueError:
            pass
            
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


# LLM-generated content at query #16
#--------------------------

# Unit test for function load
def test_load():
    import tempfile
    from pathlib import Path

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdirname:
        replay_dir = Path(tmpdirname)
        template_name = "test_template"
        context = {"cookiecutter": {"key": "value"}}

        # Write test data to a file
        dump(replay_dir, template_name, context)

        # Load the data and assert it matches the original context
        loaded_context = load(replay_dir, template_name)
        assert loaded_context == context, "Loaded context does not match the original context"

        # Test loading a non-existent file should raise an error
        try:
            load(replay_dir, "non_existent_template")
            assert False, "Expected FileNotFoundError"
        except FileNotFoundError:
            pass

        # Test loading a file with missing 'cookiecutter' key
        invalid_context = {"key": "value"}
        invalid_file = get_file_name(replay_dir, "invalid_template")
        with open(invalid_file, "w", encoding="utf-8") as f:
            json.dump(invalid_context, f)

        try:
            load(replay_dir, "invalid_template")
            assert False, "Expected ValueError due to missing 'cookiecutter' key"
        except ValueError:
            pass


# LLM-generated content at query #17
#--------------------------

# Unit test for function load
def test_load():
    import tempfile
    from pathlib import Path

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdirname:
        replay_dir = Path(tmpdirname)
        template_name = "test_template"
        context = {"cookiecutter": {"project_name": "Test Project"}}

        # Save context to a file
        dump(replay_dir, template_name, context)

        # Load context from the file
        loaded_context = load(replay_dir, template_name)

        # Assert that the loaded context matches the original context
        assert loaded_context == context



# LLM-generated content at query #18
#--------------------------

# Unit test for function dump
def test_dump():
    import tempfile
    from pathlib import Path

    replay_dir = Path(tempfile.mkdtemp())
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}

    dump(replay_dir, template_name, context)
    replay_file = replay_dir / f"{template_name}.json"

    assert replay_file.exists()
    with open(replay_file, encoding="utf-8") as infile:
        loaded_context = json.load(infile)
    assert loaded_context == context



# LLM-generated content at query #19
#--------------------------

# Unit test for function load
def test_load():
    """Test the load function."""
    import tempfile
    from pathlib import Path

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "test_template"
        context = {"cookiecutter": {"test_key": "test_value"}}

        # Save the context to a file
        dump(replay_dir, template_name, context)

        # Load the context from the file
        loaded_context = load(replay_dir, template_name)

        # Check if the loaded context matches the original
        assert loaded_context == context

        # Check if the file has the correct name
        expected_file = replay_dir / f"{template_name}.json"
        assert expected_file.exists()


# LLM-generated content at query #20
#--------------------------

# Unit test for function dump
def test_dump():
    import tempfile
    import shutil
    import json

    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()

    try:
        # Define test data
        template_name = "example_template"
        context = {"cookiecutter": {"project_name": "Test Project"}}

        # Call the dump function
        dump(temp_dir, template_name, context)

        # Verify the file was created
        file_path = get_file_name(temp_dir, template_name)
        assert os.path.exists(file_path)

        # Verify the content of the file
        with open(file_path, "r", encoding="utf-8") as file:
            file_content = json.load(file)
            assert file_content == context

    finally:
        # Clean up the temporary directory
        shutil.rmtree(temp_dir)



# LLM-generated content at query #21
#--------------------------

# Unit test for function load
def test_load():
    import tempfile
    from pathlib import Path

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdirname:
        replay_dir = Path(tmpdirname)
        template_name = "test_template"
        context = {"cookiecutter": {"key": "value"}}

        # Write the context to a file
        dump(replay_dir, template_name, context)

        # Load the context from the file
        loaded_context = load(replay_dir, template_name)

        # Assert that the loaded context matches the original context
        assert loaded_context == context

        # Test loading a non-existent file
        try:
            load(replay_dir, "non_existent_template")
        except (FileNotFoundError, ValueError):
            pass
        else:
            raise AssertionError("Expected FileNotFoundError or ValueError")

        # Test loading a file with missing 'cookiecutter' key
        invalid_context = {"key": "value"}
        invalid_file = replay_dir / "invalid_template.json"
        with open(invalid_file, "w", encoding="utf-8") as f:
            json.dump(invalid_context, f)

        try:
            load(replay_dir, "invalid_template")
        except ValueError:
            pass
        else:
            raise AssertionError("Expected ValueError")



# LLM-generated content at query #22
#--------------------------

# Unit test for function dump
def test_dump():
    import tempfile
    import shutil

    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()

    try:
        template_name = "test_template"
        context = {"cookiecutter": {"key1": "value1", "key2": "value2"}}

        # Call the dump function
        dump(temp_dir, template_name, context)

        # Verify the file was created
        file_path = get_file_name(temp_dir, template_name)
        assert os.path.exists(file_path)

        # Verify the content of the file
        with open(file_path, encoding="utf-8") as infile:
            loaded_context = json.load(infile)
        assert loaded_context == context

    finally:
        # Clean up the temporary directory
        shutil.rmtree(temp_dir)



# LLM-generated content at query #23
#--------------------------

# Unit test for function load
def test_load():
    replay_dir = "test_replay"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}

    dump(replay_dir, template_name, context)
    loaded_context = load(replay_dir, template_name)
    
    assert loaded_context == context

    os.remove(get_file_name(replay_dir, template_name))
    os.rmdir(replay_dir)


# LLM-generated content at query #24
#--------------------------

# Unit test for function dump
def test_dump():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp_dir:
        replay_dir = Path(tmp_dir)
        template_name = "test_template"
        context = {"cookiecutter": {"key": "value"}}

        dump(replay_dir, template_name, context)

        expected_file = replay_dir / f"{template_name}.json"
        assert expected_file.exists()

        with open(expected_file, 'r', encoding="utf-8") as file:
            loaded_context = json.load(file)
            assert loaded_context == context



# LLM-generated content at query #25
#--------------------------

# Unit test for function dump
def test_dump():
    """Test the dump function."""
    import tempfile
    import shutil
    from pathlib import Path

    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    replay_dir = Path(temp_dir)

    # Test data
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}

    try:
        # Call the function
        dump(replay_dir, template_name, context)

        # Verify the file was created
        expected_file = replay_dir / f"{template_name}.json"
        assert expected_file.exists()

        # Verify the content
        with open(expected_file, encoding="utf-8") as f:
            loaded_context = json.load(f)
        assert loaded_context == context

    finally:
        # Clean up
        shutil.rmtree(temp_dir)


# LLM-generated content at query #26
#--------------------------

# Unit test for function load
def test_load():
    """Test the load function."""
    import tempfile
    from pathlib import Path

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "test_template"
        test_context = {"cookiecutter": {"key": "value"}}

        # Test file not found
        try:
            load(replay_dir, template_name)
            assert False, "Expected ValueError"
        except FileNotFoundError:
            pass

        # Test valid load
        dump(replay_dir, template_name, test_context)
        loaded_context = load(replay_dir, template_name)
        assert loaded_context == test_context

        # Test invalid context (no cookiecutter key)
        invalid_file = replay_dir / "invalid.json"
        invalid_file.write_text('{"invalid": "data"}')
        try:
            load(replay_dir, "invalid")
            assert False, "Expected ValueError"
        except ValueError as e:
            assert str(e) == "Context is required to contain a cookiecutter key"


# LLM-generated content at query #27
#--------------------------

# Unit test for function dump
def test_dump():
    """Test the dump function."""
    import tempfile
    import shutil
    from pathlib import Path

    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    replay_dir = Path(temp_dir)
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}

    try:
        # Test dump function
        dump(replay_dir, template_name, context)

        # Verify the file was created
        expected_file = replay_dir / f"{template_name}.json"
        assert expected_file.exists()

        # Verify the content
        with open(expected_file, 'r', encoding="utf-8") as f:
            loaded_context = json.load(f)
        assert loaded_context == context

    finally:
        # Clean up
        shutil.rmtree(temp_dir)


# LLM-generated content at query #28
#--------------------------

# Unit test for function load
def test_load():
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    
    # Ensure the replay directory exists
    make_sure_path_exists(replay_dir)
    
    # Dump the context to a file
    dump(replay_dir, template_name, context)
    
    # Load the context from the file
    loaded_context = load(replay_dir, template_name)
    
    # Assert that the loaded context matches the original context
    assert loaded_context == context
    
    # Clean up by removing the replay file and directory
    replay_file = get_file_name(replay_dir, template_name)
    os.remove(replay_file)
    os.rmdir(replay_dir)



# LLM-generated content at query #29
#--------------------------

# Unit test for function dump
def test_dump():
    """Test the dump function."""
    import tempfile
    import shutil

    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    try:
        # Test data
        template_name = "test_template"
        context = {"cookiecutter": {"key": "value"}}

        # Call the function
        dump(temp_dir, template_name, context)

        # Verify the file was created
        expected_file = os.path.join(temp_dir, f"{template_name}.json")
        assert os.path.exists(expected_file)

        # Verify the content
        with open(expected_file, 'r', encoding="utf-8") as f:
            loaded_context = json.load(f)
        assert loaded_context == context

    finally:
        # Clean up
        shutil.rmtree(temp_dir)


# LLM-generated content at query #30
#--------------------------

# Unit test for function dump
def test_dump():
    """Test the dump function."""
    import tempfile
    import shutil

    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    try:
        # Test data
        template_name = "test_template"
        context = {"cookiecutter": {"key": "value"}}

        # Call the function
        dump(temp_dir, template_name, context)

        # Verify the file was created
        file_path = os.path.join(temp_dir, f"{template_name}.json")
        assert os.path.exists(file_path)

        # Verify the content
        with open(file_path, encoding="utf-8") as f:
            content = json.load(f)
        assert content == context

    finally:
        # Clean up
        shutil.rmtree(temp_dir)


# LLM-generated content at query #31
#--------------------------

# Unit test for function load
def test_load():
    import tempfile
    from pathlib import Path

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        replay_dir = Path(tmp_dir)
        template_name = "test_template"
        context = {"cookiecutter": {"key": "value"}}

        # Write the context to a file
        dump(replay_dir, template_name, context)

        # Load the context from the file
        loaded_context = load(replay_dir, template_name)

        # Assert that the loaded context matches the original context
        assert loaded_context == context


# LLM-generated content at query #32
#--------------------------

# Unit test for function load
def test_load():
    replay_dir = "test_replay_dir"
    template_name = "test_template"
    expected_context = {"cookiecutter": {"key": "value"}}

    # Ensure replay_dir exists
    os.makedirs(replay_dir, exist_ok=True)

    # Create a test replay file
    replay_file = os.path.join(replay_dir, f"{template_name}.json")
    with open(replay_file, "w", encoding="utf-8") as outfile:
        json.dump(expected_context, outfile, indent=2)

    # Test that load reads the correct context
    loaded_context = load(replay_dir, template_name)
    assert loaded_context == expected_context

    # Clean up
    os.remove(replay_file)
    os.rmdir(replay_dir)


# LLM-generated content at query #33
#--------------------------

# Unit test for function dump
def test_dump():
    import tempfile
    import shutil
    from pathlib import Path

    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    replay_dir = Path(temp_dir)
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}

    try:
        # Test dump function
        dump(replay_dir, template_name, context)
        
        # Verify file was created
        expected_file = replay_dir / f"{template_name}.json"
        assert expected_file.exists(), "File was not created"
        
        # Verify file content
        with open(expected_file, 'r', encoding="utf-8") as f:
            content = json.load(f)
        assert content == context, "File content does not match context"
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


# LLM-generated content at query #34
#--------------------------

# Unit test for function load
def test_load():
    # Setup
    replay_dir = "tests/replay"
    template_name = "template"
    context = {"cookiecutter": {"key": "value"}}
    replay_file = os.path.join(replay_dir, template_name + ".json")

    # Ensure replay directory exists
    os.makedirs(replay_dir, exist_ok=True)

    # Write test context to file
    with open(replay_file, "w", encoding="utf-8") as outfile:
        json.dump(context, outfile)

    # Test load function
    loaded_context = load(replay_dir, template_name)

    # Assertions
    assert loaded_context == context

    # Cleanup
    os.remove(replay_file)
    os.rmdir(replay_dir)


# LLM-generated content at query #35
#--------------------------

# Unit test for function dump
def test_dump():
    replay_dir = "test_replay"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    
    dump(replay_dir, template_name, context)
    
    assert os.path.exists(os.path.join(replay_dir, f"{template_name}.json"))
    
    with open(os.path.join(replay_dir, f"{template_name}.json"), "r", encoding="utf-8") as file:
        loaded_context = json.load(file)
    
    assert loaded_context == context



