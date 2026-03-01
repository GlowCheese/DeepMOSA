####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_get_user_config():
    # Test 1: Return default config when default_config=True
    result = get_user_config(default_config=True)
    assert result == DEFAULT_CONFIG
    
    # Test 2: Merge custom dict with defaults when default_config is dict
    custom_config = {
        'cookiecutters_dir': '/custom/cookiecutters/',
        'abbreviations': {'custom': 'https://custom.com/{0}'}
    }
    result = get_user_config(default_config=custom_config)
    assert result['cookiecutters_dir'] == '/custom/cookiecutters/'
    assert result['abbreviations']['custom'] == 'https://custom.com/{0}'
    assert result['abbreviations']['gh'] == 'https://github.com/{0}.git'  # Preserved from defaults
    assert result['replay_dir'] == DEFAULT_CONFIG['replay_dir']  # Unchanged from defaults
    
    # Test 3: Load from custom config file path
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""
cookiecutters_dir: /tmp/test_cookiecutters/
replay_dir: /tmp/test_replay/
default_context:
  key1: value1
abbreviations:
  test: https://test.com/{0}
""")
        config_file = f.name
    
    try:
        result = get_user_config(config_file=config_file)
        assert result['cookiecutters_dir'] == os.path.expanduser('/tmp/test_cookiecutters/')
        assert result['replay_dir'] == os.path.expanduser('/tmp/test_replay/')
        assert result['default_context']['key1'] == 'value1'
        assert result['abbreviations']['test'] == 'https://test.com/{0}'
        assert result['abbreviations']['gh'] == 'https://github.com/{0}.git'  # Preserved from defaults
    finally:
        os.unlink(config_file)
    
    # Test 4: Load from environment variable when set
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""
cookiecutters_dir: /env/cookiecutters/
replay_dir: /env/replay/
""")
        env_config_file = f.name
    
    os.environ['COOKIECUTTER_CONFIG'] = env_config_file
    try:
        result = get_user_config()
        assert result['cookiecutters_dir'] == os.path.expanduser('/env/cookiecutters/')
        assert result['replay_dir'] == os.path.expanduser('/env/replay/')
    finally:
        del os.environ['COOKIECUTTER_CONFIG']
        os.unlink(env_config_file)
    
    # Test 5: Return defaults when no config exists
    # Temporarily rename user config if it exists
    user_config_backup = None
    if os.path.exists(USER_CONFIG_PATH):
        user_config_backup = USER_CONFIG_PATH + '.backup'
        os.rename(USER_CONFIG_PATH, user_config_backup)
    
    try:
        # Also ensure environment variable is not set
        if 'COOKIECUTTER_CONFIG' in os.environ:
            del os.environ['COOKIECUTTER_CONFIG']
        
        result = get_user_config()
        assert result == DEFAULT_CONFIG
    finally:
        # Restore user config if it existed
        if user_config_backup and os.path.exists(user_config_backup):
            os.rename(user_config_backup, USER_CONFIG_PATH)
    
    # Test 6: Invalid YAML in config file raises InvalidConfiguration
    import tempfile
    import pytest
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("invalid: yaml: [")
        invalid_file = f.name
    
    try:
        with pytest.raises(InvalidConfiguration):
            get_user_config(config_file=invalid_file)
    finally:
        os.unlink(invalid_file)
    
    # Test 7: Non-dict YAML raises InvalidConfiguration
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("just a string")
        invalid_file = f.name
    
    try:
        with pytest.raises(InvalidConfiguration):
            get_user_config(config_file=invalid_file)
    finally:
        os.unlink(invalid_file)
    
    # Test 8: Non-existent config file raises ConfigDoesNotExistException
    import pytest
    
    with pytest.raises(ConfigDoesNotExistException):
        get_user_config(config_file='/non/existent/path/config.yaml')


# LLM-generated content at query #2
#--------------------------

```python
def test_get_config():
    import tempfile
    import os
    from pathlib import Path
    import pytest
    from cookiecutter.exceptions import ConfigDoesNotExistException, InvalidConfiguration

    # Test 1: Config file does not exist
    with pytest.raises(ConfigDoesNotExistException) as exc_info:
        get_config("/non/existent/path")
    assert "Config file /non/existent/path does not exist." in str(exc_info.value)

    # Test 2: Valid YAML config with custom values
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""
cookiecutters_dir: /custom/cookiecutters
replay_dir: /custom/replay
default_context:
  key1: value1
abbreviations:
  custom: https://custom.com/{0}
""")
        config_file = f.name

    try:
        config = get_config(config_file)
        assert config['cookiecutters_dir'] == '/custom/cookiecutters'
        assert config['replay_dir'] == '/custom/replay'
        assert config['default_context'] == {'key1': 'value1'}
        assert config['abbreviations'] == {
            'gh': 'https://github.com/{0}.git',
            'gl': 'https://gitlab.com/{0}.git',
            'bb': 'https://bitbucket.org/{0}',
            'custom': 'https://custom.com/{0}'
        }
    finally:
        os.unlink(config_file)

    # Test 3: Invalid YAML format
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("invalid: yaml: [")
        config_file = f.name

    try:
        with pytest.raises(InvalidConfiguration) as exc_info:
            get_config(config_file)
        assert f"Unable to parse YAML file {config_file}." in str(exc_info.value)
    finally:
        os.unlink(config_file)

    # Test 4: YAML top-level is not a dict
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("- item1\n- item2")
        config_file = f.name

    try:
        with pytest.raises(InvalidConfiguration) as exc_info:
            get_config(config_file)
        assert f"Top-level element of YAML file {config_file} should be an object." in str(exc_info.value)
    finally:
        os.unlink(config_file)

    # Test 5: Empty YAML file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("")
        config_file = f.name

    try:
        config = get_config(config_file)
        assert config['cookiecutters_dir'] == os.path.expanduser('~/.cookiecutters/')
        assert config['replay_dir'] == os.path.expanduser('~/.cookiecutter_replay/')
        assert config['default_context'] == {}
        assert config['abbreviations'] == {
            'gh': 'https://github.com/{0}.git',
            'gl': 'https://gitlab.com/{0}.git',
            'bb': 'https://bitbucket.org/{0}'
        }
    finally:
        os.unlink(config_file)

    # Test 6: Path expansion in config values
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""
cookiecutters_dir: ~/custom_cookiecutters
replay_dir: $HOME/custom_replay
""")
        config_file = f.name

    try:
        config = get_config(config_file)
        expected_cookiecutters = os.path.expanduser('~/custom_cookiecutters')
        expected_replay = os.path.expandvars('$HOME/custom_replay')
        assert config['cookiecutters_dir'] == expected_cookiecutters
        assert config['replay_dir'] == expected_replay
    finally:
        os.unlink(config_file)

    # Test 7: Partial config - should merge with defaults
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""
replay_dir: /partial/replay
""")
        config_file = f.name

    try:
        config = get_config(config_file)
        assert config['cookiecutters_dir'] == os.path.expanduser('~/.cookiecutters/')
        assert config['replay_dir'] == '/partial/replay'
        assert config['abbreviations'] == {
            'gh': 'https://github.com/{0}.git',
            'gl': 'https://gitlab.com/{0}.git',
            'bb': 'https://bitbucket.org/{0}'
        }
    finally:
        os.unlink(config_file)

    # Test 8: Config with nested dict merging
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""
abbreviations:
  custom1: https://custom1.com/{0}
  gh: https://custom.github.com/{0}.git
""")
        config_file = f.name

    try:
        config = get_config(config_file)
        assert config['abbreviations'] == {
            'gh': 'https://custom.github.com/{0}.git',
            'gl': 'https://gitlab.com/{0}.git',
            'bb': 'https://bitbucket.org/{0}',
            'custom1': 'https://custom1.com/{0}'
        }
    finally:
        os.unlink(config_file)


# LLM-generated content at query #3
#--------------------------

```python
def test_get_config():
    import tempfile
    import os
    from pathlib import Path
    from cookiecutter.exceptions import ConfigDoesNotExistException, InvalidConfiguration

    # Test 1: Config file does not exist
    try:
        get_config("/non/existent/path")
        assert False, "Expected ConfigDoesNotExistException"
    except ConfigDoesNotExistException as e:
        assert "does not exist" in str(e)

    # Test 2: Invalid YAML format
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("invalid: yaml: [")
        temp_path = f.name
    
    try:
        get_config(temp_path)
        assert False, "Expected InvalidConfiguration"
    except InvalidConfiguration as e:
        assert "Unable to parse YAML" in str(e)
    finally:
        os.unlink(temp_path)

    # Test 3: YAML is not a dict
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("- item1\n- item2")
        temp_path = f.name
    
    try:
        get_config(temp_path)
        assert False, "Expected InvalidConfiguration"
    except InvalidConfiguration as e:
        assert "should be an object" in str(e)
    finally:
        os.unlink(temp_path)

    # Test 4: Valid YAML with custom values
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""
cookiecutters_dir: /custom/cookiecutters
replay_dir: /custom/replay
default_context:
  key1: value1
abbreviations:
  custom: https://custom.com/{0}
""")
        temp_path = f.name
    
    config = get_config(temp_path)
    assert config['cookiecutters_dir'] == os.path.expanduser('/custom/cookiecutters')
    assert config['replay_dir'] == os.path.expanduser('/custom/replay')
    assert config['default_context'] == {'key1': 'value1'}
    assert config['abbreviations']['custom'] == 'https://custom.com/{0}'
    assert config['abbreviations']['gh'] == 'https://github.com/{0}.git'
    
    os.unlink(temp_path)

    # Test 5: Path expansion with environment variables and user home
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""
cookiecutters_dir: ~/test_cookiecutters
replay_dir: $HOME/test_replay
""")
        temp_path = f.name
    
    config = get_config(temp_path)
    assert config['cookiecutters_dir'] == os.path.expanduser('~/test_cookiecutters')
    assert config['replay_dir'] == os.path.expanduser('~/test_replay')
    
    os.unlink(temp_path)

    # Test 6: Empty YAML file returns defaults
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("")
        temp_path = f.name
    
    config = get_config(temp_path)
    assert config['cookiecutters_dir'] == os.path.expanduser('~/.cookiecutters/')
    assert config['replay_dir'] == os.path.expanduser('~/.cookiecutter_replay/')
    assert config['default_context'] == {}
    assert config['abbreviations'] == BUILTIN_ABBREVIATIONS
    
    os.unlink(temp_path)

    # Test 7: YAML with None value returns defaults
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("null")
        temp_path = f.name
    
    config = get_config(temp_path)
    assert config['cookiecutters_dir'] == os.path.expanduser('~/.cookiecutters/')
    assert config['replay_dir'] == os.path.expanduser('~/.cookiecutter_replay/')
    
    os.unlink(temp_path)

    # Test 8: Deep merge preserves nested dicts
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""
abbreviations:
  new: https://new.com/{0}
""")
        temp_path = f.name
    
    config = get_config(temp_path)
    assert config['abbreviations']['new'] == 'https://new.com/{0}'
    assert config['abbreviations']['gh'] == 'https://github.com/{0}.git'
    assert config['abbreviations']['gl'] == 'https://gitlab.com/{0}.git'
    assert config['abbreviations']['bb'] == 'https://bitbucket.org/{0}'
    
    os.unlink(temp_path)

    # Test 9: Path object as input
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("replay_dir: /test/path")
        temp_path = f.name
    
    config = get_config(Path(temp_path))
    assert config['replay_dir'] == os.path.expanduser('/test/path')
    
    os.unlink(temp_path)


# LLM-generated content at query #4
#--------------------------

```python
def test_get_config():
    import tempfile
    import os
    from pathlib import Path
    from cookiecutter.exceptions import ConfigDoesNotExistException, InvalidConfiguration

    # Test 1: Config file does not exist
    try:
        get_config("/non/existent/path")
        assert False, "Should have raised ConfigDoesNotExistException"
    except ConfigDoesNotExistException as e:
        assert "does not exist" in str(e)

    # Test 2: Valid config file with minimal content
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("cookiecutters_dir: /custom/cookiecutters\n")
        config_file = f.name

    try:
        config = get_config(config_file)
        assert config['cookiecutters_dir'] == '/custom/cookiecutters'
        assert config['replay_dir'] == os.path.expanduser('~/.cookiecutter_replay/')
        assert config['default_context'] == {}
        assert config['abbreviations'] == {
            'gh': 'https://github.com/{0}.git',
            'gl': 'https://gitlab.com/{0}.git',
            'bb': 'https://bitbucket.org/{0}',
        }
    finally:
        os.unlink(config_file)

    # Test 3: Valid config file with all fields and path expansion
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""cookiecutters_dir: $HOME/custom_cookiecutters
replay_dir: ~/custom_replay
default_context:
  author: Test Author
abbreviations:
  custom: https://custom.com/{0}
""")
        config_file = f.name

    try:
        config = get_config(config_file)
        assert config['cookiecutters_dir'] == os.path.expanduser('~/custom_cookiecutters')
        assert config['replay_dir'] == os.path.expanduser('~/custom_replay')
        assert config['default_context']['author'] == 'Test Author'
        assert config['abbreviations']['custom'] == 'https://custom.com/{0}'
        assert config['abbreviations']['gh'] == 'https://github.com/{0}.git'  # Built-in preserved
    finally:
        os.unlink(config_file)

    # Test 4: Invalid YAML
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("invalid: yaml: :")
        config_file = f.name

    try:
        try:
            get_config(config_file)
            assert False, "Should have raised InvalidConfiguration"
        except InvalidConfiguration as e:
            assert "Unable to parse YAML" in str(e)
    finally:
        os.unlink(config_file)

    # Test 5: YAML is not a dict (list)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("- item1\n- item2")
        config_file = f.name

    try:
        try:
            get_config(config_file)
            assert False, "Should have raised InvalidConfiguration"
        except InvalidConfiguration as e:
            assert "Top-level element" in str(e) and "should be an object" in str(e)
    finally:
        os.unlink(config_file)

    # Test 6: YAML is not a dict (scalar)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("just a string")
        config_file = f.name

    try:
        try:
            get_config(config_file)
            assert False, "Should have raised InvalidConfiguration"
        except InvalidConfiguration as e:
            assert "Top-level element" in str(e) and "should be an object" in str(e)
    finally:
        os.unlink(config_file)

    # Test 7: Empty YAML file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("")
        config_file = f.name

    try:
        config = get_config(config_file)
        assert config['cookiecutters_dir'] == os.path.expanduser('~/.cookiecutters/')
        assert config['replay_dir'] == os.path.expanduser('~/.cookiecutter_replay/')
    finally:
        os.unlink(config_file)

    # Test 8: Path as Path object
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("cookiecutters_dir: /test/path")
        config_file = f.name

    try:
        config = get_config(Path(config_file))
        assert config['cookiecutters_dir'] == '/test/path'
    finally:
        os.unlink(config_file)

    # Test 9: Deep merge of nested dictionaries (abbreviations)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""abbreviations:
  gl: https://gitlab.custom.com/{0}
  new: https://new.com/{0}
""")
        config_file = f.name

    try:
        config = get_config(config_file)
        assert config['abbreviations']['gl'] == 'https://gitlab.custom.com/{0}'  # Overridden
        assert config['abbreviations']['gh'] == 'https://github.com/{0}.git'  # Preserved
        assert config['abbreviations']['new'] == 'https://new.com/{0}'  # Added
        assert config['abbreviations']['bb'] == 'https://bitbucket.org/{0}'  # Preserved
    finally:
        os.unlink(config_file)


# LLM-generated content at query #5
#--------------------------

```python
def test_get_config():
    import tempfile
    import os
    from pathlib import Path
    import pytest
    from cookiecutter.exceptions import ConfigDoesNotExistException, InvalidConfiguration

    # Test 1: Config file does not exist
    with pytest.raises(ConfigDoesNotExistException):
        get_config('/non/existent/path/config.yaml')

    # Test 2: Valid YAML config with custom values
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""
cookiecutters_dir: /custom/cookiecutters/
replay_dir: /custom/replay/
default_context:
  project_name: Test Project
abbreviations:
  custom: https://custom.com/{0}
""")
        config_path = f.name

    try:
        config = get_config(config_path)
        assert config['cookiecutters_dir'] == '/custom/cookiecutters/'
        assert config['replay_dir'] == '/custom/replay/'
        assert config['default_context']['project_name'] == 'Test Project'
        assert config['abbreviations']['custom'] == 'https://custom.com/{0}'
        # Verify built-in abbreviations are preserved
        assert config['abbreviations']['gh'] == 'https://github.com/{0}.git'
    finally:
        os.unlink(config_path)

    # Test 3: YAML with environment variables and user expansion
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""
cookiecutters_dir: ~/test_cookiecutters/
replay_dir: $HOME/test_replay/
""")
        config_path = f.name

    try:
        config = get_config(config_path)
        expected_cookiecutters = os.path.expanduser('~/test_cookiecutters/')
        expected_replay = os.path.expandvars('$HOME/test_replay/')
        assert config['cookiecutters_dir'] == expected_cookiecutters
        assert config['replay_dir'] == expected_replay
    finally:
        os.unlink(config_path)

    # Test 4: Invalid YAML format
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("invalid: yaml: [")
        config_path = f.name

    try:
        with pytest.raises(InvalidConfiguration):
            get_config(config_path)
    finally:
        os.unlink(config_path)

    # Test 5: YAML top-level is not a dict (list)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("- item1\n- item2")
        config_path = f.name

    try:
        with pytest.raises(InvalidConfiguration):
            get_config(config_path)
    finally:
        os.unlink(config_path)

    # Test 6: Empty YAML file (returns empty dict)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("")
        config_path = f.name

    try:
        config = get_config(config_path)
        assert config['cookiecutters_dir'] == os.path.expanduser('~/.cookiecutters/')
        assert config['replay_dir'] == os.path.expanduser('~/.cookiecutter_replay/')
        assert config['abbreviations'] == {
            'gh': 'https://github.com/{0}.git',
            'gl': 'https://gitlab.com/{0}.git',
            'bb': 'https://bitbucket.org/{0}',
        }
    finally:
        os.unlink(config_path)

    # Test 7: YAML with nested dict merging
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""
abbreviations:
  custom1: https://custom1.com/{0}
  gh: https://github.com/custom/{0}.git
""")
        config_path = f.name

    try:
        config = get_config(config_path)
        assert config['abbreviations']['custom1'] == 'https://custom1.com/{0}'
        assert config['abbreviations']['gh'] == 'https://github.com/custom/{0}.git'
        assert config['abbreviations']['gl'] == 'https://gitlab.com/{0}.git'
        assert config['abbreviations']['bb'] == 'https://bitbucket.org/{0}'
    finally:
        os.unlink(config_path)

    # Test 8: Path object as input
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("cookiecutters_dir: /test/path/")
        config_path = f.name

    try:
        config = get_config(Path(config_path))
        assert config['cookiecutters_dir'] == '/test/path/'
    finally:
        os.unlink(config_path)


# LLM-generated content at query #6
#--------------------------

```python
def test_get_config():
    import tempfile
    import os
    from pathlib import Path
    import pytest
    from cookiecutter.exceptions import ConfigDoesNotExistException, InvalidConfiguration

    # Test 1: Config file does not exist
    with pytest.raises(ConfigDoesNotExistException):
        get_config("/non/existent/path/config.yaml")

    # Test 2: Valid YAML config with custom values
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""
cookiecutters_dir: /custom/cookiecutters
replay_dir: /custom/replay
default_context:
  key1: value1
abbreviations:
  custom: https://custom.com/{0}
""")
        config_path = f.name

    try:
        config = get_config(config_path)
        assert config['cookiecutters_dir'] == '/custom/cookiecutters'
        assert config['replay_dir'] == '/custom/replay'
        assert config['default_context'] == {'key1': 'value1'}
        assert config['abbreviations'] == {
            'gh': 'https://github.com/{0}.git',
            'gl': 'https://gitlab.com/{0}.git',
            'bb': 'https://bitbucket.org/{0}',
            'custom': 'https://custom.com/{0}'
        }
    finally:
        os.unlink(config_path)

    # Test 3: YAML with environment variables and user home expansion
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""
cookiecutters_dir: $HOME/custom_cookiecutters
replay_dir: ~/custom_replay
""")
        config_path = f.name

    try:
        config = get_config(config_path)
        home = os.path.expanduser('~')
        assert config['cookiecutters_dir'] == os.path.join(home, 'custom_cookiecutters')
        assert config['replay_dir'] == os.path.join(home, 'custom_replay')
    finally:
        os.unlink(config_path)

    # Test 4: Invalid YAML format
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("invalid: yaml: [")
        config_path = f.name

    try:
        with pytest.raises(InvalidConfiguration):
            get_config(config_path)
    finally:
        os.unlink(config_path)

    # Test 5: YAML top-level element is not a dict (list)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("- item1\n- item2")
        config_path = f.name

    try:
        with pytest.raises(InvalidConfiguration):
            get_config(config_path)
    finally:
        os.unlink(config_path)

    # Test 6: YAML top-level element is not a dict (scalar)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("just a string")
        config_path = f.name

    try:
        with pytest.raises(InvalidConfiguration):
            get_config(config_path)
    finally:
        os.unlink(config_path)

    # Test 7: Empty YAML file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("")
        config_path = f.name

    try:
        config = get_config(config_path)
        assert config['cookiecutters_dir'] == os.path.expanduser('~/.cookiecutters/')
        assert config['replay_dir'] == os.path.expanduser('~/.cookiecutter_replay/')
        assert config['default_context'] == {}
        assert config['abbreviations'] == {
            'gh': 'https://github.com/{0}.git',
            'gl': 'https://gitlab.com/{0}.git',
            'bb': 'https://bitbucket.org/{0}'
        }
    finally:
        os.unlink(config_path)

    # Test 8: YAML with only partial config
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("replay_dir: /partial/replay")
        config_path = f.name

    try:
        config = get_config(config_path)
        assert config['cookiecutters_dir'] == os.path.expanduser('~/.cookiecutters/')
        assert config['replay_dir'] == '/partial/replay'
        assert config['default_context'] == {}
        assert config['abbreviations'] == {
            'gh': 'https://github.com/{0}.git',
            'gl': 'https://gitlab.com/{0}.git',
            'bb': 'https://bitbucket.org/{0}'
        }
    finally:
        os.unlink(config_path)

    # Test 9: Path object as input
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("replay_dir: /path/object/test")
        config_path = f.name

    try:
        config = get_config(Path(config_path))
        assert config['replay_dir'] == '/path/object/test'
    finally:
        os.unlink(config_path)


# LLM-generated content at query #7
#--------------------------

```python
def test_get_config():
    import tempfile
    import os
    from pathlib import Path
    from cookiecutter.exceptions import ConfigDoesNotExistException, InvalidConfiguration

    # Test 1: Config file does not exist
    with tempfile.TemporaryDirectory() as tmpdir:
        non_existent = Path(tmpdir) / "nonexistent.yaml"
        try:
            get_config(non_existent)
            assert False, "Should have raised ConfigDoesNotExistException"
        except ConfigDoesNotExistException as e:
            assert str(non_existent) in str(e)

    # Test 2: Valid config file with custom values
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""
cookiecutters_dir: /custom/cookiecutters
replay_dir: /custom/replay
default_context:
  project_name: Test Project
abbreviations:
  custom: https://custom.com/{0}
""")
        config_file = f.name

    try:
        config = get_config(config_file)
        assert config['cookiecutters_dir'] == '/custom/cookiecutters'
        assert config['replay_dir'] == '/custom/replay'
        assert config['default_context']['project_name'] == 'Test Project'
        assert config['abbreviations']['custom'] == 'https://custom.com/{0}'
        # Built-in abbreviations should still be present
        assert config['abbreviations']['gh'] == 'https://github.com/{0}.git'
    finally:
        os.unlink(config_file)

    # Test 3: Config file with environment variables and user expansion
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""
cookiecutters_dir: $HOME/custom_cookiecutters
replay_dir: ~/custom_replay
""")
        config_file = f.name

    try:
        config = get_config(config_file)
        assert config['cookiecutters_dir'] == os.path.expanduser(os.path.expandvars('$HOME/custom_cookiecutters'))
        assert config['replay_dir'] == os.path.expanduser('~/custom_replay')
    finally:
        os.unlink(config_file)

    # Test 4: Invalid YAML file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("invalid: yaml: [")
        config_file = f.name

    try:
        try:
            get_config(config_file)
            assert False, "Should have raised InvalidConfiguration"
        except InvalidConfiguration as e:
            assert "Unable to parse YAML" in str(e)
    finally:
        os.unlink(config_file)

    # Test 5: YAML file with non-dict top-level element
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("just a string")
        config_file = f.name

    try:
        try:
            get_config(config_file)
            assert False, "Should have raised InvalidConfiguration"
        except InvalidConfiguration as e:
            assert "Top-level element" in str(e)
    finally:
        os.unlink(config_file)

    # Test 6: Empty YAML file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("")
        config_file = f.name

    try:
        config = get_config(config_file)
        # Should return default config
        assert config['cookiecutters_dir'] == os.path.expanduser('~/.cookiecutters/')
        assert config['replay_dir'] == os.path.expanduser('~/.cookiecutter_replay/')
    finally:
        os.unlink(config_file)

    # Test 7: Config file with only some values specified
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""
replay_dir: /only/replay
""")
        config_file = f.name

    try:
        config = get_config(config_file)
        assert config['replay_dir'] == '/only/replay'
        # Other values should be defaults
        assert config['cookiecutters_dir'] == os.path.expanduser('~/.cookiecutters/')
        assert config['abbreviations'] == BUILTIN_ABBREVIATIONS
    finally:
        os.unlink(config_file)

    # Test 8: Deep merge of nested dictionaries
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""
abbreviations:
  custom: https://custom.com/{0}
  gh: https://github.com/custom/{0}.git
""")
        config_file = f.name

    try:
        config = get_config(config_file)
        # Custom abbreviation added
        assert config['abbreviations']['custom'] == 'https://custom.com/{0}'
        # Built-in abbreviation overridden
        assert config['abbreviations']['gh'] == 'https://github.com/custom/{0}.git'
        # Other built-in abbreviations preserved
        assert config['abbreviations']['gl'] == 'https://gitlab.com/{0}.git'
        assert config['abbreviations']['bb'] == 'https://bitbucket.org/{0}'
    finally:
        os.unlink(config_file)


# LLM-generated content at query #8
#--------------------------

```python
def test_get_config():
    import tempfile
    import os
    from pathlib import Path
    from cookiecutter.exceptions import ConfigDoesNotExistException, InvalidConfiguration

    # Test 1: Config file does not exist
    with tempfile.TemporaryDirectory() as tmpdir:
        non_existent = Path(tmpdir) / "nonexistent.yaml"
        try:
            get_config(non_existent)
            assert False, "Should raise ConfigDoesNotExistException"
        except ConfigDoesNotExistException as e:
            assert str(non_existent) in str(e)

    # Test 2: Valid YAML config with custom values
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""
cookiecutters_dir: /custom/cookiecutters
replay_dir: /custom/replay
default_context:
  key1: value1
abbreviations:
  custom: https://custom.com/{0}
""")
        config_file = f.name

    try:
        config = get_config(config_file)
        assert config['cookiecutters_dir'] == '/custom/cookiecutters'
        assert config['replay_dir'] == '/custom/replay'
        assert config['default_context'] == {'key1': 'value1'}
        assert config['abbreviations'] == {
            'gh': 'https://github.com/{0}.git',
            'gl': 'https://gitlab.com/{0}.git',
            'bb': 'https://bitbucket.org/{0}',
            'custom': 'https://custom.com/{0}'
        }
    finally:
        os.unlink(config_file)

    # Test 3: Invalid YAML format
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("invalid: yaml: [")
        config_file = f.name

    try:
        try:
            get_config(config_file)
            assert False, "Should raise InvalidConfiguration"
        except InvalidConfiguration as e:
            assert 'Unable to parse YAML' in str(e)
    finally:
        os.unlink(config_file)

    # Test 4: YAML top-level is not a dict
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("- item1\n- item2")
        config_file = f.name

    try:
        try:
            get_config(config_file)
            assert False, "Should raise InvalidConfiguration"
        except InvalidConfiguration as e:
            assert 'Top-level element' in str(e)
    finally:
        os.unlink(config_file)

    # Test 5: Empty YAML file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("")
        config_file = f.name

    try:
        config = get_config(config_file)
        assert config['cookiecutters_dir'] == os.path.expanduser('~/.cookiecutters/')
        assert config['replay_dir'] == os.path.expanduser('~/.cookiecutter_replay/')
        assert config['default_context'] == {}
        assert config['abbreviations'] == {
            'gh': 'https://github.com/{0}.git',
            'gl': 'https://gitlab.com/{0}.git',
            'bb': 'https://bitbucket.org/{0}'
        }
    finally:
        os.unlink(config_file)

    # Test 6: Path expansion in config
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""
cookiecutters_dir: $HOME/custom_cookiecutters
replay_dir: ~/custom_replay
""")
        config_file = f.name

    try:
        config = get_config(config_file)
        expected_cookiecutters = os.path.expandvars('$HOME/custom_cookiecutters')
        expected_replay = os.path.expanduser('~/custom_replay')
        assert config['cookiecutters_dir'] == expected_cookiecutters
        assert config['replay_dir'] == expected_replay
    finally:
        os.unlink(config_file)

    # Test 7: Partial config merges with defaults
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""
replay_dir: /partial/replay
""")
        config_file = f.name

    try:
        config = get_config(config_file)
        assert config['cookiecutters_dir'] == os.path.expanduser('~/.cookiecutters/')
        assert config['replay_dir'] == '/partial/replay'
        assert config['abbreviations'] == {
            'gh': 'https://github.com/{0}.git',
            'gl': 'https://gitlab.com/{0}.git',
            'bb': 'https://bitbucket.org/{0}'
        }
    finally:
        os.unlink(config_file)


# LLM-generated content at query #9
#--------------------------

```python
def test_get_config():
    import tempfile
    import os
    from pathlib import Path
    from cookiecutter.exceptions import ConfigDoesNotExistException, InvalidConfiguration

    # Test 1: Config file does not exist
    try:
        get_config("/non/existent/path")
        assert False, "Should have raised ConfigDoesNotExistException"
    except ConfigDoesNotExistException as e:
        assert "does not exist" in str(e)

    # Test 2: Valid YAML config with custom values
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""
cookiecutters_dir: /custom/cookiecutters
replay_dir: /custom/replay
default_context:
  project_name: Test Project
abbreviations:
  custom: https://custom.com/{0}
""")
        config_file = f.name

    try:
        config = get_config(config_file)
        assert config['cookiecutters_dir'] == '/custom/cookiecutters'
        assert config['replay_dir'] == '/custom/replay'
        assert config['default_context']['project_name'] == 'Test Project'
        assert config['abbreviations']['custom'] == 'https://custom.com/{0}'
        # Built-in abbreviations should still exist
        assert config['abbreviations']['gh'] == 'https://github.com/{0}.git'
    finally:
        os.unlink(config_file)

    # Test 3: Invalid YAML file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("invalid: yaml: [")
        config_file = f.name

    try:
        try:
            get_config(config_file)
            assert False, "Should have raised InvalidConfiguration"
        except InvalidConfiguration as e:
            assert "Unable to parse YAML" in str(e)
    finally:
        os.unlink(config_file)

    # Test 4: YAML file with non-dict top-level element
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("- item1\n- item2")
        config_file = f.name

    try:
        try:
            get_config(config_file)
            assert False, "Should have raised InvalidConfiguration"
        except InvalidConfiguration as e:
            assert "should be an object" in str(e)
    finally:
        os.unlink(config_file)

    # Test 5: Empty YAML file (should use defaults)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("")
        config_file = f.name

    try:
        config = get_config(config_file)
        assert config['cookiecutters_dir'] == os.path.expanduser('~/.cookiecutters/')
        assert config['replay_dir'] == os.path.expanduser('~/.cookiecutter_replay/')
        assert config['default_context'] == {}
        assert config['abbreviations'] == {
            'gh': 'https://github.com/{0}.git',
            'gl': 'https://gitlab.com/{0}.git',
            'bb': 'https://bitbucket.org/{0}',
        }
    finally:
        os.unlink(config_file)

    # Test 6: Path expansion in config
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""
replay_dir: $HOME/custom_replay
cookiecutters_dir: ~/custom_cookiecutters
""")
        config_file = f.name

    try:
        config = get_config(config_file)
        assert config['replay_dir'] == os.path.expanduser('~/custom_replay')
        assert config['cookiecutters_dir'] == os.path.expanduser('~/custom_cookiecutters')
    finally:
        os.unlink(config_file)

    # Test 7: Nested dictionary merging
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""
abbreviations:
  custom: https://custom.com/{0}
  gh: https://github.custom.com/{0}.git
""")
        config_file = f.name

    try:
        config = get_config(config_file)
        # Custom abbreviation added
        assert config['abbreviations']['custom'] == 'https://custom.com/{0}'
        # Built-in abbreviation overridden
        assert config['abbreviations']['gh'] == 'https://github.custom.com/{0}.git'
        # Other built-in abbreviations preserved
        assert config['abbreviations']['gl'] == 'https://gitlab.com/{0}.git'
        assert config['abbreviations']['bb'] == 'https://bitbucket.org/{0}'
    finally:
        os.unlink(config_file)

    # Test 8: Config with Path object as input
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("replay_dir: /test/path")
        config_file = f.name

    try:
        config = get_config(Path(config_file))
        assert config['replay_dir'] == '/test/path'
    finally:
        os.unlink(config_file)


# LLM-generated content at query #10
#--------------------------

```python
def test_get_config():
    import tempfile
    import os
    from pathlib import Path
    from cookiecutter.exceptions import ConfigDoesNotExistException, InvalidConfiguration

    # Test 1: Config file does not exist
    try:
        get_config("/non/existent/path")
        assert False, "Expected ConfigDoesNotExistException"
    except ConfigDoesNotExistException as e:
        assert "Config file /non/existent/path does not exist." in str(e)

    # Test 2: Valid config file with custom values
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""
cookiecutters_dir: /custom/cookiecutters
replay_dir: /custom/replay
default_context:
  key1: value1
abbreviations:
  custom: https://custom.com/{0}
""")
        config_file = f.name

    try:
        config = get_config(config_file)
        assert config['cookiecutters_dir'] == '/custom/cookiecutters'
        assert config['replay_dir'] == '/custom/replay'
        assert config['default_context'] == {'key1': 'value1'}
        assert config['abbreviations']['custom'] == 'https://custom.com/{0}'
        assert config['abbreviations']['gh'] == 'https://github.com/{0}.git'
    finally:
        os.unlink(config_file)

    # Test 3: Config file with environment variables and user home expansion
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""
cookiecutters_dir: $HOME/test_cookiecutters
replay_dir: ~/test_replay
""")
        config_file = f.name

    try:
        config = get_config(config_file)
        assert config['cookiecutters_dir'] == os.path.expanduser('~/test_cookiecutters')
        assert config['replay_dir'] == os.path.expanduser('~/test_replay')
    finally:
        os.unlink(config_file)

    # Test 4: Invalid YAML format
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("invalid: yaml: content")
        config_file = f.name

    try:
        get_config(config_file)
        assert False, "Expected InvalidConfiguration"
    except InvalidConfiguration as e:
        assert f"Unable to parse YAML file {config_file}." in str(e)
    finally:
        os.unlink(config_file)

    # Test 5: YAML file with non-dict top-level element
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("- item1\n- item2")
        config_file = f.name

    try:
        get_config(config_file)
        assert False, "Expected InvalidConfiguration"
    except InvalidConfiguration as e:
        assert f"Top-level element of YAML file {config_file} should be an object." in str(e)
    finally:
        os.unlink(config_file)

    # Test 6: Empty YAML file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("")
        config_file = f.name

    try:
        config = get_config(config_file)
        assert config['cookiecutters_dir'] == os.path.expanduser('~/.cookiecutters/')
        assert config['replay_dir'] == os.path.expanduser('~/.cookiecutter_replay/')
        assert config['default_context'] == {}
        assert config['abbreviations'] == {
            'gh': 'https://github.com/{0}.git',
            'gl': 'https://gitlab.com/{0}.git',
            'bb': 'https://bitbucket.org/{0}',
        }
    finally:
        os.unlink(config_file)

    # Test 7: Path object as input
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("cookiecutters_dir: /test/path")
        config_file = f.name

    try:
        config = get_config(Path(config_file))
        assert config['cookiecutters_dir'] == '/test/path'
    finally:
        os.unlink(config_file)


# LLM-generated content at query #11
#--------------------------

```python
def test_get_config():
    import tempfile
    import os
    from pathlib import Path
    from cookiecutter.exceptions import ConfigDoesNotExistException, InvalidConfiguration

    # Test 1: Config file does not exist
    with tempfile.TemporaryDirectory() as tmpdir:
        non_existent_path = Path(tmpdir) / "nonexistent.yaml"
        try:
            get_config(non_existent_path)
            assert False, "Should have raised ConfigDoesNotExistException"
        except ConfigDoesNotExistException as e:
            assert str(non_existent_path) in str(e)

    # Test 2: Valid YAML config with custom values
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""
cookiecutters_dir: "~/custom_cookiecutters"
replay_dir: "~/custom_replay"
default_context:
    author_name: "Test Author"
abbreviations:
    custom: "https://custom.com/{0}"
""")
        config_path = f.name

    try:
        config = get_config(config_path)
        assert config['cookiecutters_dir'] == os.path.expanduser("~/custom_cookiecutters")
        assert config['replay_dir'] == os.path.expanduser("~/custom_replay")
        assert config['default_context']['author_name'] == "Test Author"
        assert config['abbreviations']['custom'] == "https://custom.com/{0}"
        # Built-in abbreviations should still be present
        assert config['abbreviations']['gh'] == "https://github.com/{0}.git"
    finally:
        os.unlink(config_path)

    # Test 3: Invalid YAML format
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("invalid: yaml: [")
        config_path = f.name

    try:
        try:
            get_config(config_path)
            assert False, "Should have raised InvalidConfiguration"
        except InvalidConfiguration as e:
            assert "Unable to parse YAML" in str(e)
    finally:
        os.unlink(config_path)

    # Test 4: YAML top-level is not a dict
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("- item1\n- item2")
        config_path = f.name

    try:
        try:
            get_config(config_path)
            assert False, "Should have raised InvalidConfiguration"
        except InvalidConfiguration as e:
            assert "Top-level element" in str(e)
    finally:
        os.unlink(config_path)

    # Test 5: Empty YAML file (should return defaults)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("")
        config_path = f.name

    try:
        config = get_config(config_path)
        assert config['cookiecutters_dir'] == os.path.expanduser("~/.cookiecutters/")
        assert config['replay_dir'] == os.path.expanduser("~/.cookiecutter_replay/")
        assert config['default_context'] == {}
        assert config['abbreviations'] == BUILTIN_ABBREVIATIONS
    finally:
        os.unlink(config_path)

    # Test 6: Path expansion with environment variables
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""
cookiecutters_dir: "$HOME/env_cookiecutters"
replay_dir: "~/user_replay"
""")
        config_path = f.name

    try:
        config = get_config(config_path)
        expected_cookie_dir = os.path.expandvars("$HOME/env_cookiecutters")
        expected_cookie_dir = os.path.expanduser(expected_cookie_dir)
        assert config['cookiecutters_dir'] == expected_cookie_dir
        assert config['replay_dir'] == os.path.expanduser("~/user_replay")
    finally:
        os.unlink(config_path)

    # Test 7: Partial config (should merge with defaults)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""
replay_dir: "~/partial_replay"
""")
        config_path = f.name

    try:
        config = get_config(config_path)
        assert config['cookiecutters_dir'] == os.path.expanduser("~/.cookiecutters/")
        assert config['replay_dir'] == os.path.expanduser("~/partial_replay")
        assert config['abbreviations'] == BUILTIN_ABBREVIATIONS
    finally:
        os.unlink(config_path)


# LLM-generated content at query #12
#--------------------------

```python
def test_get_config():
    import tempfile
    import os
    from pathlib import Path
    import pytest
    from cookiecutter.exceptions import ConfigDoesNotExistException, InvalidConfiguration

    # Test 1: Config file does not exist
    with pytest.raises(ConfigDoesNotExistException) as exc_info:
        get_config("/non/existent/path/config.yaml")
    assert "Config file /non/existent/path/config.yaml does not exist." in str(exc_info.value)

    # Test 2: Valid YAML config with expansions
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""
cookiecutters_dir: "~/test_cookiecutters"
replay_dir: "$HOME/test_replay"
default_context:
  key1: value1
abbreviations:
  custom: "https://custom.com/{0}"
""")
        config_file = f.name

    try:
        config = get_config(config_file)
        assert config['cookiecutters_dir'] == os.path.expanduser('~/test_cookiecutters')
        assert config['replay_dir'] == os.path.expanduser(os.path.expandvars('$HOME/test_replay'))
        assert config['default_context'] == {'key1': 'value1'}
        assert config['abbreviations'] == {
            'gh': 'https://github.com/{0}.git',
            'gl': 'https://gitlab.com/{0}.git',
            'bb': 'https://bitbucket.org/{0}',
            'custom': 'https://custom.com/{0}'
        }
    finally:
        os.unlink(config_file)

    # Test 3: Invalid YAML format
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("invalid: yaml: [")
        config_file = f.name

    try:
        with pytest.raises(InvalidConfiguration) as exc_info:
            get_config(config_file)
        assert f"Unable to parse YAML file {config_file}." in str(exc_info.value)
    finally:
        os.unlink(config_file)

    # Test 4: YAML top-level is not a dict (list)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("- item1\n- item2")
        config_file = f.name

    try:
        with pytest.raises(InvalidConfiguration) as exc_info:
            get_config(config_file)
        assert f"Top-level element of YAML file {config_file} should be an object." in str(exc_info.value)
    finally:
        os.unlink(config_file)

    # Test 5: YAML top-level is not a dict (scalar)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("just a string")
        config_file = f.name

    try:
        with pytest.raises(InvalidConfiguration) as exc_info:
            get_config(config_file)
        assert f"Top-level element of YAML file {config_file} should be an object." in str(exc_info.value)
    finally:
        os.unlink(config_file)

    # Test 6: Empty YAML file (returns empty dict)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("")
        config_file = f.name

    try:
        config = get_config(config_file)
        assert config['cookiecutters_dir'] == os.path.expanduser('~/.cookiecutters/')
        assert config['replay_dir'] == os.path.expanduser('~/.cookiecutter_replay/')
        assert config['default_context'] == {}
        assert config['abbreviations'] == BUILTIN_ABBREVIATIONS
    finally:
        os.unlink(config_file)

    # Test 7: YAML with None value (treated as empty dict)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("null")
        config_file = f.name

    try:
        config = get_config(config_file)
        assert config['cookiecutters_dir'] == os.path.expanduser('~/.cookiecutters/')
        assert config['replay_dir'] == os.path.expanduser('~/.cookiecutter_replay/')
        assert config['default_context'] == {}
        assert config['abbreviations'] == BUILTIN_ABBREVIATIONS
    finally:
        os.unlink(config_file)

    # Test 8: Path object as input
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""
cookiecutters_dir: "~/test_path"
replay_dir: "~/test_replay_path"
""")
        config_file = f.name

    try:
        config = get_config(Path(config_file))
        assert config['cookiecutters_dir'] == os.path.expanduser('~/test_path')
        assert config['replay_dir'] == os.path.expanduser('~/test_replay_path')
    finally:
        os.unlink(config_file)

    # Test 9: Deep merge of nested dictionaries
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""
default_context:
  new_key: new_value
abbreviations:
  gl: "https://overridden.gitlab.com/{0}.git"
  new_abbr: "https://new.com/{0}"
""")
        config_file = f.name

    try:
        config = get_config(config_file)
        assert config['default_context'] == {'new_key': 'new_value'}
        assert config['abbreviations']['gh'] == 'https://github.com/{0}.git'  # preserved
        assert config['abbreviations']['gl'] == 'https://overridden.gitlab.com/{0}.git'  # overridden
        assert config['abbreviations']['bb'] == 'https://bitbucket.org/{0}'  # preserved
        assert config['abbreviations']['new_abbr'] == 'https://new.com/{0}'  # added
    finally:
        os.unlink(config_file)


# LLM-generated content at query #13
#--------------------------

```python
def test_get_config():
    import tempfile
    import os
    from pathlib import Path
    import pytest
    from cookiecutter.exceptions import ConfigDoesNotExistException, InvalidConfiguration

    # Test 1: Config file does not exist
    with pytest.raises(ConfigDoesNotExistException) as exc_info:
        get_config("/non/existent/path/config.yaml")
    assert "Config file /non/existent/path/config.yaml does not exist." in str(exc_info.value)

    # Test 2: Valid YAML config with custom values
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""
cookiecutters_dir: /custom/cookiecutters/
replay_dir: /custom/replay/
default_context:
    key1: value1
    key2: value2
abbreviations:
    custom: https://custom.com/{0}
""")
        config_file = f.name

    try:
        config = get_config(config_file)
        assert config['cookiecutters_dir'] == '/custom/cookiecutters/'
        assert config['replay_dir'] == '/custom/replay/'
        assert config['default_context'] == {'key1': 'value1', 'key2': 'value2'}
        assert config['abbreviations'] == {
            'gh': 'https://github.com/{0}.git',
            'gl': 'https://gitlab.com/{0}.git',
            'bb': 'https://bitbucket.org/{0}',
            'custom': 'https://custom.com/{0}'
        }
    finally:
        os.unlink(config_file)

    # Test 3: YAML parsing error
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("invalid: yaml: [")
        config_file = f.name

    try:
        with pytest.raises(InvalidConfiguration) as exc_info:
            get_config(config_file)
        assert f"Unable to parse YAML file {config_file}." in str(exc_info.value)
    finally:
        os.unlink(config_file)

    # Test 4: YAML top-level element is not a dict
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("- item1\n- item2")
        config_file = f.name

    try:
        with pytest.raises(InvalidConfiguration) as exc_info:
            get_config(config_file)
        assert f"Top-level element of YAML file {config_file} should be an object." in str(exc_info.value)
    finally:
        os.unlink(config_file)

    # Test 5: Empty YAML file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("")
        config_file = f.name

    try:
        config = get_config(config_file)
        assert config['cookiecutters_dir'] == os.path.expanduser('~/.cookiecutters/')
        assert config['replay_dir'] == os.path.expanduser('~/.cookiecutter_replay/')
        assert config['default_context'] == {}
        assert config['abbreviations'] == {
            'gh': 'https://github.com/{0}.git',
            'gl': 'https://gitlab.com/{0}.git',
            'bb': 'https://bitbucket.org/{0}'
        }
    finally:
        os.unlink(config_file)

    # Test 6: Path expansion in config
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""
cookiecutters_dir: ~/custom_cookiecutters/
replay_dir: $HOME/custom_replay/
""")
        config_file = f.name

    try:
        config = get_config(config_file)
        expected_cookiecutters_dir = os.path.expanduser('~/custom_cookiecutters/')
        expected_replay_dir = os.path.expandvars('$HOME/custom_replay/')
        assert config['cookiecutters_dir'] == expected_cookiecutters_dir
        assert config['replay_dir'] == expected_replay_dir
    finally:
        os.unlink(config_file)

    # Test 7: Partial config - only some values specified
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""
replay_dir: /partial/replay/
""")
        config_file = f.name

    try:
        config = get_config(config_file)
        assert config['cookiecutters_dir'] == os.path.expanduser('~/.cookiecutters/')
        assert config['replay_dir'] == '/partial/replay/'
        assert config['default_context'] == {}
        assert config['abbreviations'] == {
            'gh': 'https://github.com/{0}.git',
            'gl': 'https://gitlab.com/{0}.git',
            'bb': 'https://bitbucket.org/{0}'
        }
    finally:
        os.unlink(config_file)

    # Test 8: Nested default_context preservation
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""
default_context:
    project_name: "Test Project"
    author: "Test Author"
""")
        config_file = f.name

    try:
        config = get_config(config_file)
        assert config['default_context'] == {
            'project_name': 'Test Project',
            'author': 'Test Author'
        }
    finally:
        os.unlink(config_file)

    # Test 9: Path object as input
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("replay_dir: /test/path/")
        config_file = f.name

    try:
        config = get_config(Path(config_file))
        assert config['replay_dir'] == '/test/path/'
    finally:
        os.unlink(config_file)


# LLM-generated content at query #14
#--------------------------

```python
def test_get_config():
    import tempfile
    import os
    from pathlib import Path
    from cookiecutter.exceptions import ConfigDoesNotExistException, InvalidConfiguration

    # Test 1: Config file does not exist
    with tempfile.TemporaryDirectory() as tmpdir:
        non_existent_path = Path(tmpdir) / "nonexistent.yaml"
        try:
            get_config(non_existent_path)
            assert False, "Should have raised ConfigDoesNotExistException"
        except ConfigDoesNotExistException as e:
            assert str(non_existent_path) in str(e)

    # Test 2: Invalid YAML file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("invalid: yaml: :")
        invalid_yaml_path = f.name

    try:
        try:
            get_config(invalid_yaml_path)
            assert False, "Should have raised InvalidConfiguration"
        except InvalidConfiguration as e:
            assert "Unable to parse YAML" in str(e)
    finally:
        os.unlink(invalid_yaml_path)

    # Test 3: YAML file with non-dict top-level element
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("- item1\n- item2")
        list_yaml_path = f.name

    try:
        try:
            get_config(list_yaml_path)
            assert False, "Should have raised InvalidConfiguration"
        except InvalidConfiguration as e:
            assert "should be an object" in str(e)
    finally:
        os.unlink(list_yaml_path)

    # Test 4: Empty YAML file (should load defaults)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("")
        empty_yaml_path = f.name

    try:
        config = get_config(empty_yaml_path)
        assert config['cookiecutters_dir'] == os.path.expanduser('~/.cookiecutters/')
        assert config['replay_dir'] == os.path.expanduser('~/.cookiecutter_replay/')
        assert config['default_context'] == {}
        assert config['abbreviations'] == BUILTIN_ABBREVIATIONS
    finally:
        os.unlink(empty_yaml_path)

    # Test 5: Valid YAML file with custom values
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""cookiecutters_dir: /custom/cookiecutters
replay_dir: /custom/replay
default_context:
  key1: value1
abbreviations:
  custom: https://custom.com/{0}
""")
        custom_yaml_path = f.name

    try:
        config = get_config(custom_yaml_path)
        assert config['cookiecutters_dir'] == '/custom/cookiecutters'
        assert config['replay_dir'] == '/custom/replay'
        assert config['default_context'] == {'key1': 'value1'}
        assert config['abbreviations'] == {
            'gh': 'https://github.com/{0}.git',
            'gl': 'https://gitlab.com/{0}.git',
            'bb': 'https://bitbucket.org/{0}',
            'custom': 'https://custom.com/{0}'
        }
    finally:
        os.unlink(custom_yaml_path)

    # Test 6: Path expansion in config values
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""cookiecutters_dir: $HOME/custom_cookiecutters
replay_dir: ~/custom_replay
""")
        expand_yaml_path = f.name

    try:
        config = get_config(expand_yaml_path)
        assert config['cookiecutters_dir'] == os.path.expandvars('$HOME/custom_cookiecutters')
        assert config['replay_dir'] == os.path.expanduser('~/custom_replay')
    finally:
        os.unlink(expand_yaml_path)

    # Test 7: Partial config (should merge with defaults)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""replay_dir: /partial/replay
default_context:
  partial_key: partial_value
""")
        partial_yaml_path = f.name

    try:
        config = get_config(partial_yaml_path)
        assert config['cookiecutters_dir'] == os.path.expanduser('~/.cookiecutters/')
        assert config['replay_dir'] == '/partial/replay'
        assert config['default_context'] == {'partial_key': 'partial_value'}
        assert config['abbreviations'] == BUILTIN_ABBREVIATIONS
    finally:
        os.unlink(partial_yaml_path)

    # Test 8: Config with nested dict merging
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""abbreviations:
  custom1: https://custom1.com/{0}
  gh: https://custom.github.com/{0}.git
""")
        nested_yaml_path = f.name

    try:
        config = get_config(nested_yaml_path)
        assert config['abbreviations']['gh'] == 'https://custom.github.com/{0}.git'
        assert config['abbreviations']['gl'] == 'https://gitlab.com/{0}.git'
        assert config['abbreviations']['bb'] == 'https://bitbucket.org/{0}'
        assert config['abbreviations']['custom1'] == 'https://custom1.com/{0}'
    finally:
        os.unlink(nested_yaml_path)


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_get_config():
    import tempfile
    import os
    from pathlib import Path
    from cookiecutter.exceptions import ConfigDoesNotExistException, InvalidConfiguration

    # Test 1: Config file does not exist
    with tempfile.TemporaryDirectory() as tmpdir:
        non_existent = Path(tmpdir) / "nonexistent.yaml"
        try:
            get_config(non_existent)
            assert False, "Should have raised ConfigDoesNotExistException"
        except ConfigDoesNotExistException as e:
            assert str(non_existent) in str(e)

    # Test 2: Valid config file with custom values
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""
cookiecutters_dir: /custom/cookiecutters
replay_dir: /custom/replay
default_context:
  key1: value1
abbreviations:
  custom: https://custom.com/{0}
""")
        config_file = f.name

    try:
        config = get_config(config_file)
        assert config['cookiecutters_dir'] == '/custom/cookiecutters'
        assert config['replay_dir'] == '/custom/replay'
        assert config['default_context'] == {'key1': 'value1'}
        assert config['abbreviations'] == {
            'gh': 'https://github.com/{0}.git',
            'gl': 'https://gitlab.com/{0}.git',
            'bb': 'https://bitbucket.org/{0}',
            'custom': 'https://custom.com/{0}'
        }
    finally:
        os.unlink(config_file)

    # Test 3: Config file with environment variables and user expansion
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""
cookiecutters_dir: ~/test_cookiecutters
replay_dir: $HOME/test_replay
""")
        config_file = f.name

    try:
        config = get_config(config_file)
        expected_replay = os.path.expanduser(os.path.expandvars('$HOME/test_replay'))
        expected_cookies = os.path.expanduser('~/test_cookiecutters')
        assert config['replay_dir'] == expected_replay
        assert config['cookiecutters_dir'] == expected_cookies
    finally:
        os.unlink(config_file)

    # Test 4: Invalid YAML
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("invalid: yaml: [")
        config_file = f.name

    try:
        try:
            get_config(config_file)
            assert False, "Should have raised InvalidConfiguration"
        except InvalidConfiguration as e:
            assert "Unable to parse YAML" in str(e)
    finally:
        os.unlink(config_file)

    # Test 5: YAML top-level is not a dict
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("- item1\n- item2")
        config_file = f.name

    try:
        try:
            get_config(config_file)
            assert False, "Should have raised InvalidConfiguration"
        except InvalidConfiguration as e:
            assert "Top-level element" in str(e)
    finally:
        os.unlink(config_file)

    # Test 6: Empty YAML file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("")
        config_file = f.name

    try:
        config = get_config(config_file)
        assert config['cookiecutters_dir'] == os.path.expanduser('~/.cookiecutters/')
        assert config['replay_dir'] == os.path.expanduser('~/.cookiecutter_replay/')
        assert config['default_context'] == {}
        assert config['abbreviations'] == {
            'gh': 'https://github.com/{0}.git',
            'gl': 'https://gitlab.com/{0}.git',
            'bb': 'https://bitbucket.org/{0}',
        }
    finally:
        os.unlink(config_file)

    # Test 7: Config with nested dict merging
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""
abbreviations:
  gh: https://custom.github.com/{0}.git
  new: https://new.com/{0}
""")
        config_file = f.name

    try:
        config = get_config(config_file)
        assert config['abbreviations']['gh'] == 'https://custom.github.com/{0}.git'
        assert config['abbreviations']['gl'] == 'https://gitlab.com/{0}.git'
        assert config['abbreviations']['bb'] == 'https://bitbucket.org/{0}'
        assert config['abbreviations']['new'] == 'https://new.com/{0}'
    finally:
        os.unlink(config_file)


# LLM-generated content at query #2
#--------------------------

```python
def test_get_config():
    import tempfile
    import os
    from pathlib import Path
    from cookiecutter.exceptions import ConfigDoesNotExistException, InvalidConfiguration

    # Test 1: Config file does not exist
    with tempfile.TemporaryDirectory() as tmpdir:
        non_existent = Path(tmpdir) / "nonexistent.yaml"
        try:
            get_config(non_existent)
            assert False, "Should have raised ConfigDoesNotExistException"
        except ConfigDoesNotExistException as e:
            assert str(non_existent) in str(e)

    # Test 2: Invalid YAML format
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("invalid: yaml: :")
        temp_path = f.name
    
    try:
        try:
            get_config(temp_path)
            assert False, "Should have raised InvalidConfiguration"
        except InvalidConfiguration as e:
            assert "Unable to parse YAML" in str(e)
    finally:
        os.unlink(temp_path)

    # Test 3: YAML is not a dict
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("- item1\n- item2")
        temp_path = f.name
    
    try:
        try:
            get_config(temp_path)
            assert False, "Should have raised InvalidConfiguration"
        except InvalidConfiguration as e:
            assert "Top-level element" in str(e)
    finally:
        os.unlink(temp_path)

    # Test 4: Empty YAML file (should return defaults)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("")
        temp_path = f.name
    
    try:
        result = get_config(temp_path)
        assert result['cookiecutters_dir'] == os.path.expanduser('~/.cookiecutters/')
        assert result['replay_dir'] == os.path.expanduser('~/.cookiecutter_replay/')
        assert result['abbreviations'] == {
            'gh': 'https://github.com/{0}.git',
            'gl': 'https://gitlab.com/{0}.git',
            'bb': 'https://bitbucket.org/{0}',
        }
        assert result['default_context'] == {}
    finally:
        os.unlink(temp_path)

    # Test 5: Valid YAML with custom values
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""cookiecutters_dir: /custom/cookiecutters
replay_dir: /custom/replay
default_context:
  key1: value1
abbreviations:
  custom: https://custom.com/{0}
""")
        temp_path = f.name
    
    try:
        result = get_config(temp_path)
        assert result['cookiecutters_dir'] == '/custom/cookiecutters'
        assert result['replay_dir'] == '/custom/replay'
        assert result['default_context'] == {'key1': 'value1'}
        assert result['abbreviations'] == {
            'gh': 'https://github.com/{0}.git',
            'gl': 'https://gitlab.com/{0}.git',
            'bb': 'https://bitbucket.org/{0}',
            'custom': 'https://custom.com/{0}'
        }
    finally:
        os.unlink(temp_path)

    # Test 6: Path expansion in config values
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""cookiecutters_dir: $HOME/test_cookiecutters
replay_dir: ~/test_replay
""")
        temp_path = f.name
    
    try:
        result = get_config(temp_path)
        expected_cookiecutters = os.path.expandvars('$HOME/test_cookiecutters')
        expected_cookiecutters = os.path.expanduser(expected_cookiecutters)
        expected_replay = os.path.expanduser('~/test_replay')
        
        assert result['cookiecutters_dir'] == expected_cookiecutters
        assert result['replay_dir'] == expected_replay
    finally:
        os.unlink(temp_path)

    # Test 7: Partial config (should merge with defaults)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""cookiecutters_dir: /partial/cookiecutters
default_context:
  project_name: Test
""")
        temp_path = f.name
    
    try:
        result = get_config(temp_path)
        assert result['cookiecutters_dir'] == '/partial/cookiecutters'
        assert result['replay_dir'] == os.path.expanduser('~/.cookiecutter_replay/')
        assert result['default_context'] == {'project_name': 'Test'}
        assert 'gh' in result['abbreviations']
    finally:
        os.unlink(temp_path)


# LLM-generated content at query #3
#--------------------------

```python
def test_get_config():
    import tempfile
    import os
    from pathlib import Path
    from cookiecutter.exceptions import ConfigDoesNotExistException, InvalidConfiguration

    # Test 1: Config file does not exist
    with tempfile.TemporaryDirectory() as tmpdir:
        non_existent_path = Path(tmpdir) / "nonexistent.yaml"
        try:
            get_config(non_existent_path)
            assert False, "Expected ConfigDoesNotExistException"
        except ConfigDoesNotExistException as e:
            assert str(non_existent_path) in str(e)

    # Test 2: Valid YAML config with custom values
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""
cookiecutters_dir: /custom/cookiecutters
replay_dir: /custom/replay
default_context:
  key1: value1
  key2: value2
abbreviations:
  custom: https://custom.com/{0}
""")
        config_path = f.name

    try:
        config = get_config(config_path)
        assert config['cookiecutters_dir'] == os.path.expanduser('/custom/cookiecutters')
        assert config['replay_dir'] == os.path.expanduser('/custom/replay')
        assert config['default_context'] == {'key1': 'value1', 'key2': 'value2'}
        assert config['abbreviations']['custom'] == 'https://custom.com/{0}'
        assert config['abbreviations']['gh'] == 'https://github.com/{0}.git'
    finally:
        os.unlink(config_path)

    # Test 3: YAML with environment variables and user home expansion
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""
cookiecutters_dir: $HOME/test_cookiecutters
replay_dir: ~/test_replay
""")
        config_path = f.name

    try:
        config = get_config(config_path)
        expected_cookiecutters = os.path.expanduser(os.path.expandvars('$HOME/test_cookiecutters'))
        expected_replay = os.path.expanduser('~/test_replay')
        assert config['cookiecutters_dir'] == expected_cookiecutters
        assert config['replay_dir'] == expected_replay
    finally:
        os.unlink(config_path)

    # Test 4: Invalid YAML syntax
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""
invalid: [yaml: syntax
""")
        config_path = f.name

    try:
        try:
            get_config(config_path)
            assert False, "Expected InvalidConfiguration"
        except InvalidConfiguration as e:
            assert "Unable to parse YAML" in str(e)
    finally:
        os.unlink(config_path)

    # Test 5: YAML top-level is not a dict (list)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("- item1\n- item2\n")
        config_path = f.name

    try:
        try:
            get_config(config_path)
            assert False, "Expected InvalidConfiguration"
        except InvalidConfiguration as e:
            assert "Top-level element of YAML file" in str(e)
    finally:
        os.unlink(config_path)

    # Test 6: YAML top-level is not a dict (string)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("just a string")
        config_path = f.name

    try:
        try:
            get_config(config_path)
            assert False, "Expected InvalidConfiguration"
        except InvalidConfiguration as e:
            assert "Top-level element of YAML file" in str(e)
    finally:
        os.unlink(config_path)

    # Test 7: Empty YAML file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("")
        config_path = f.name

    try:
        config = get_config(config_path)
        assert config['cookiecutters_dir'] == os.path.expanduser('~/.cookiecutters/')
        assert config['replay_dir'] == os.path.expanduser('~/.cookiecutter_replay/')
        assert config['default_context'] == {}
        assert config['abbreviations'] == {
            'gh': 'https://github.com/{0}.git',
            'gl': 'https://gitlab.com/{0}.git',
            'bb': 'https://bitbucket.org/{0}',
        }
    finally:
        os.unlink(config_path)

    # Test 8: YAML with None value (should be treated as empty dict)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("null")
        config_path = f.name

    try:
        config = get_config(config_path)
        assert config['cookiecutters_dir'] == os.path.expanduser('~/.cookiecutters/')
        assert config['replay_dir'] == os.path.expanduser('~/.cookiecutter_replay/')
    finally:
        os.unlink(config_path)

    # Test 9: Partial config - should merge with defaults
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""
cookiecutters_dir: /partial/path
""")
        config_path = f.name

    try:
        config = get_config(config_path)
        assert config['cookiecutters_dir'] == os.path.expanduser('/partial/path')
        assert config['replay_dir'] == os.path.expanduser('~/.cookiecutter_replay/')
        assert config['abbreviations']['gh'] == 'https://github.com/{0}.git'
    finally:
        os.unlink(config_path)

    # Test 10: Nested dict merging for abbreviations
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""
abbreviations:
  custom: https://custom.com/{0}
""")
        config_path = f.name

    try:
        config = get_config(config_path)
        assert config['abbreviations']['gh'] == 'https://github.com/{0}.git'
        assert config['abbreviations']['custom'] == 'https://custom.com/{0}'
        assert len(config['abbreviations']) == 4
    finally:
        os.unlink(config_path)


# LLM-generated content at query #4
#--------------------------

```python
def test_get_config():
    import tempfile
    import os
    from pathlib import Path
    from cookiecutter.exceptions import ConfigDoesNotExistException, InvalidConfiguration
    
    # Test 1: Config file does not exist
    with tempfile.TemporaryDirectory() as tmpdir:
        non_existent_path = Path(tmpdir) / "nonexistent.yaml"
        try:
            get_config(non_existent_path)
            assert False, "Expected ConfigDoesNotExistException"
        except ConfigDoesNotExistException as e:
            assert str(non_existent_path) in str(e)
    
    # Test 2: Valid YAML config with custom values
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""
cookiecutters_dir: /custom/cookiecutters
replay_dir: /custom/replay
default_context:
  key1: value1
abbreviations:
  custom: https://custom.com/{0}
""")
        config_path = f.name
    
    try:
        config = get_config(config_path)
        assert config['cookiecutters_dir'] == '/custom/cookiecutters'
        assert config['replay_dir'] == '/custom/replay'
        assert config['default_context'] == {'key1': 'value1'}
        assert config['abbreviations']['custom'] == 'https://custom.com/{0}'
        # Built-in abbreviations should still be present
        assert config['abbreviations']['gh'] == 'https://github.com/{0}.git'
    finally:
        os.unlink(config_path)
    
    # Test 3: Invalid YAML file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("invalid: yaml: [")
        config_path = f.name
    
    try:
        try:
            get_config(config_path)
            assert False, "Expected InvalidConfiguration"
        except InvalidConfiguration as e:
            assert 'Unable to parse YAML' in str(e)
    finally:
        os.unlink(config_path)
    
    # Test 4: YAML file with non-dict top-level element
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("- item1\n- item2")
        config_path = f.name
    
    try:
        try:
            get_config(config_path)
            assert False, "Expected InvalidConfiguration"
        except InvalidConfiguration as e:
            assert 'Top-level element' in str(e)
    finally:
        os.unlink(config_path)
    
    # Test 5: YAML file with environment variables and user expansion
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""
replay_dir: $HOME/custom_replay
cookiecutters_dir: ~/custom_cookiecutters
""")
        config_path = f.name
    
    try:
        config = get_config(config_path)
        assert config['replay_dir'] == os.path.expanduser(os.path.expandvars('$HOME/custom_replay'))
        assert config['cookiecutters_dir'] == os.path.expanduser('~/custom_cookiecutters')
    finally:
        os.unlink(config_path)
    
    # Test 6: Empty YAML file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("")
        config_path = f.name
    
    try:
        config = get_config(config_path)
        # Should return default config
        assert config['cookiecutters_dir'] == os.path.expanduser('~/.cookiecutters/')
        assert config['replay_dir'] == os.path.expanduser('~/.cookiecutter_replay/')
        assert config['abbreviations'] == BUILTIN_ABBREVIATIONS
    finally:
        os.unlink(config_path)
    
    # Test 7: YAML file with only partial config
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""
cookiecutters_dir: /partial/path
""")
        config_path = f.name
    
    try:
        config = get_config(config_path)
        assert config['cookiecutters_dir'] == '/partial/path'
        # Other values should be defaults
        assert config['replay_dir'] == os.path.expanduser('~/.cookiecutter_replay/')
        assert config['abbreviations'] == BUILTIN_ABBREVIATIONS
    finally:
        os.unlink(config_path)


