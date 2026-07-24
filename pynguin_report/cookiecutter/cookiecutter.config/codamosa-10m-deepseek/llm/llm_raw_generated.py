####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

# Unit test for function get_config
def test_get_config():
    import tempfile
    import pytest

    # Test with a valid config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml') as tmp:
        tmp.write('cookiecutters_dir: /tmp/cookiecutters\n')
        tmp.write('replay_dir: /tmp/replay\n')
        tmp.flush()
        config = get_config(tmp.name)
        assert config['cookiecutters_dir'] == '/tmp/cookiecutters'
        assert config['replay_dir'] == '/tmp/replay'

    # Test with a non-existent config file
    with pytest.raises(ConfigDoesNotExistException):
        get_config('/nonexistent/path')

    # Test with an invalid YAML file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml') as tmp:
        tmp.write('invalid: yaml: file\n')
        tmp.flush()
        with pytest.raises(InvalidConfiguration):
            get_config(tmp.name)

    # Test with a YAML file that is not a dict
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml') as tmp:
        tmp.write('- item1\n- item2\n')
        tmp.flush()
        with pytest.raises(InvalidConfiguration):
            get_config(tmp.name)


# LLM-generated content at query #2
#--------------------------

# Unit test for function get_config
def test_get_config():
    # Test with a valid config file
    with open('test_config.yml', 'w') as f:
        f.write("cookiecutters_dir: ~/.cookiecutters/\nreplay_dir: ~/.cookiecutter_replay/\n")
    config = get_config('test_config.yml')
    assert config['cookiecutters_dir'] == os.path.expanduser('~/.cookiecutters/')
    assert config['replay_dir'] == os.path.expanduser('~/.cookiecutter_replay/')
    os.remove('test_config.yml')

    # Test with a non-existent config file
    try:
        get_config('non_existent_config.yml')
    except ConfigDoesNotExistException:
        pass
    else:
        assert False, "Expected ConfigDoesNotExistException"

    # Test with an invalid YAML file
    with open('invalid_config.yml', 'w') as f:
        f.write("invalid yaml")
    try:
        get_config('invalid_config.yml')
    except InvalidConfiguration:
        pass
    else:
        assert False, "Expected InvalidConfiguration"
    os.remove('invalid_config.yml')

    # Test with a YAML file that is not a dict
    with open('not_dict_config.yml', 'w') as f:
        f.write("- item1\n- item2")
    try:
        get_config('not_dict_config.yml')
    except InvalidConfiguration:
        pass
    else:
        assert False, "Expected InvalidConfiguration"
    os.remove('not_dict_config.yml')


# LLM-generated content at query #3
#--------------------------

# Unit test for function get_config
def test_get_config():
    import pytest
    from tempfile import NamedTemporaryFile
    import os

    # Test with a valid YAML file
    with NamedTemporaryFile(mode='w', delete=False) as temp_file:
        temp_file.write("cookiecutters_dir: /custom/path\n")
        temp_file.write("replay_dir: /another/path\n")
        temp_file_path = temp_file.name

    config = get_config(temp_file_path)
    assert config['cookiecutters_dir'] == '/custom/path'
    assert config['replay_dir'] == '/another/path'
    os.remove(temp_file_path)

    # Test with an invalid YAML file
    with NamedTemporaryFile(mode='w', delete=False) as temp_file:
        temp_file.write("invalid yaml content")
        temp_file_path = temp_file.name

    with pytest.raises(InvalidConfiguration):
        get_config(temp_file_path)
    os.remove(temp_file_path)

    # Test with a non-existent file
    with pytest.raises(ConfigDoesNotExistException):
        get_config('/non/existent/path')

    # Test with a YAML file that is not a dictionary
    with NamedTemporaryFile(mode='w', delete=False) as temp_file:
        temp_file.write("just a string")
        temp_file_path = temp_file.name

    with pytest.raises(InvalidConfiguration):
        get_config(temp_file_path)
    os.remove(temp_file_path)



# LLM-generated content at query #4
#--------------------------

# Unit test for function get_config
def test_get_config():
    # Test case 1: config file does not exist
    config_path = '/path/to/nonexistent/config.yaml'
    try:
        get_config(config_path)
        assert False, 'Expected ConfigDoesNotExistException'
    except ConfigDoesNotExistException:
        pass

    # Test case 2: invalid YAML file
    config_path = '/path/to/invalid/config.yaml'
    with open(config_path, 'w') as file_handle:
        file_handle.write('invalid yaml')
    try:
        get_config(config_path)
        assert False, 'Expected InvalidConfiguration'
    except InvalidConfiguration:
        pass

    # Test case 3: valid config file
    config_path = '/path/to/valid/config.yaml'
    with open(config_path, 'w') as file_handle:
        file_handle.write('cookiecutters_dir: /path/to/cookiecutters')
    config_dict = get_config(config_path)
    assert config_dict['cookiecutters_dir'] == os.path.expanduser('/path/to/cookiecutters')

    # Test case 4: config file with environment variables
    config_path = '/path/to/config_with_env.yaml'
    os.environ['TEST_ENV'] = '/path/to/test'
    with open(config_path, 'w') as file_handle:
        file_handle.write('cookiecutters_dir: $TEST_ENV/cookiecutters')
    config_dict = get_config(config_path)
    assert config_dict['cookiecutters_dir'] == os.path.expanduser('/path/to/test/cookiecutters')

    # Test case 5: config file with user home expansion
    config_path = '/path/to/config_with_home.yaml'
    with open(config_path, 'w') as file_handle:
        file_handle.write('cookiecutters_dir: ~/cookiecutters')
    config_dict = get_config(config_path)
    assert config_dict['cookiecutters_dir'] == os.path.expanduser('~/cookiecutters')

    # Test case 6: config file with nested dictionaries
    config_path = '/path/to/config_with_nested.yaml'
    with open(config_path, 'w') as file_handle:
        file_handle.write('abbreviations:\n  gh: https://github.com/{0}.git')
    config_dict = get_config(config_path)
    assert config_dict['abbreviations'] == {'gh': 'https://github.com/{0}.git'}

    # Test case 7: config file with empty YAML
    config_path = '/path/to/config_empty.yaml'
    with open(config_path, 'w') as file_handle:
        file_handle.write('')
    config_dict = get_config(config_path)
    assert config_dict == DEFAULT_CONFIG

    # Test case 8: config file with non-dict top-level element
    config_path = '/path/to/config_non_dict.yaml'
    with open(config_path, 'w') as file_handle:
        file_handle.write('- item1\n- item2')
    try:
        get_config(config_path)
        assert False, 'Expected InvalidConfiguration'
    except InvalidConfiguration:
        pass

    # Clean up test files
    os.remove('/path/to/invalid/config.yaml')
    os.remove('/path/to/valid/config.yaml')
    os.remove('/path/to/config_with_env.yaml')
    os.remove('/path/to/config_with_home.yaml')
    os.remove('/path/to/config_with_nested.yaml')
    os.remove('/path/to/config_empty.yaml')
    os.remove('/path/to/config_non_dict.yaml')


# LLM-generated content at query #5
#--------------------------

# Unit test for function get_config
def test_get_config():
    # Setup
    import tempfile
    import os
    import yaml

    # Test case 1: Valid YAML config file
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
        config_data = {
            'cookiecutters_dir': '/custom/cookiecutters',
            'replay_dir': '/custom/replay',
            'default_context': {'key': 'value'},
            'abbreviations': {'custom': 'https://custom.com/{0}.git'}
        }
        yaml.dump(config_data, temp_file)
        temp_file_path = temp_file.name

    try:
        result = get_config(temp_file_path)
        expected = {
            'cookiecutters_dir': '/custom/cookiecutters',
            'replay_dir': '/custom/replay',
            'default_context': collections.OrderedDict([('key', 'value')]),
            'abbreviations': {
                'gh': 'https://github.com/{0}.git',
                'gl': 'https://gitlab.com/{0}.git',
                'bb': 'https://bitbucket.org/{0}',
                'custom': 'https://custom.com/{0}.git'
            }
        }
        assert result == expected
    finally:
        os.remove(temp_file_path)

    # Test case 2: Invalid YAML file
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
        temp_file.write("invalid: yaml: here")
        temp_file_path = temp_file.name

    try:
        try:
            get_config(temp_file_path)
            assert False, "Expected InvalidConfiguration exception"
        except InvalidConfiguration:
            pass
    finally:
        os.remove(temp_file_path)

    # Test case 3: Non-existent file
    try:
        get_config('/non/existent/path')
        assert False, "Expected ConfigDoesNotExistException"
    except ConfigDoesNotExistException:
        pass

    # Test case 4: Top-level element is not a dict
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
        yaml.dump("not a dict", temp_file)
        temp_file_path = temp_file.name

    try:
        try:
            get_config(temp_file_path)
            assert False, "Expected InvalidConfiguration exception"
        except InvalidConfiguration:
            pass
    finally:
        os.remove(temp_file_path)


# LLM-generated content at query #6
#--------------------------

# Unit test for function get_config
def test_get_config():
    """Test the get_config function."""
    import tempfile
    import shutil
    import pytest

    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    try:
        # Create a test config file
        config_path = os.path.join(temp_dir, 'test_config.yml')
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write('cookiecutters_dir: /tmp/cookiecutters\n')
            f.write('replay_dir: /tmp/replay\n')
            f.write('default_context:\n')
            f.write('  key1: value1\n')
            f.write('abbreviations:\n')
            f.write('  gh: https://github.com/{0}.git\n')

        # Test loading the config
        config = get_config(config_path)
        assert config['cookiecutters_dir'] == '/tmp/cookiecutters'
        assert config['replay_dir'] == '/tmp/replay'
        assert config['default_context']['key1'] == 'value1'
        assert config['abbreviations']['gh'] == 'https://github.com/{0}.git'

        # Test with non-existent file
        with pytest.raises(ConfigDoesNotExistException):
            get_config('/nonexistent/path')

        # Test with invalid YAML
        invalid_config_path = os.path.join(temp_dir, 'invalid.yml')
        with open(invalid_config_path, 'w', encoding='utf-8') as f:
            f.write('invalid: yaml: file')

        with pytest.raises(InvalidConfiguration):
            get_config(invalid_config_path)

        # Test with non-dict YAML
        non_dict_config_path = os.path.join(temp_dir, 'non_dict.yml')
        with open(non_dict_config_path, 'w', encoding='utf-8') as f:
            f.write('- item1\n- item2')

        with pytest.raises(InvalidConfiguration):
            get_config(non_dict_config_path)

    finally:
        # Clean up
        shutil.rmtree(temp_dir)


# LLM-generated content at query #7
#--------------------------

# Unit test for function get_config
def test_get_config():
    from pathlib import Path
    import tempfile
    import shutil

    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    config_path = Path(temp_dir) / "test_config.yml"

    # Create a sample config file
    sample_config = {
        'cookiecutters_dir': '/custom/cookiecutters',
        'replay_dir': '/custom/replay',
        'default_context': {'key1': 'value1'},
        'abbreviations': {'custom': 'https://custom.com/{0}'},
    }
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(sample_config, f)

    # Test loading the config
    loaded_config = get_config(config_path)
    assert loaded_config['cookiecutters_dir'] == '/custom/cookiecutters'
    assert loaded_config['replay_dir'] == '/custom/replay'
    assert loaded_config['default_context'] == {'key1': 'value1'}
    assert loaded_config['abbreviations'] == {
        'gh': 'https://github.com/{0}.git',
        'gl': 'https://gitlab.com/{0}.git',
        'bb': 'https://bitbucket.org/{0}',
        'custom': 'https://custom.com/{0}',
    }

    # Clean up
    shutil.rmtree(temp_dir)


# LLM-generated content at query #8
#--------------------------

# Unit test for function get_config
def test_get_config():
    """Test the get_config function."""
    import tempfile
    import shutil
    import pytest

    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()

    try:
        # Create a config file with valid YAML
        config_path = os.path.join(temp_dir, 'valid_config.yml')
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write('cookiecutters_dir: /tmp/custom_cookiecutters\n')
            f.write('replay_dir: /tmp/custom_replay\n')
            f.write('default_context:\n')
            f.write('  key1: value1\n')
            f.write('  key2: value2\n')
            f.write('abbreviations:\n')
            f.write('  gh: "https://github.com/{0}.git"\n')

        # Test loading valid config
        config = get_config(config_path)
        assert config['cookiecutters_dir'] == '/tmp/custom_cookiecutters'
        assert config['replay_dir'] == '/tmp/custom_replay'
        assert config['default_context']['key1'] == 'value1'
        assert config['default_context']['key2'] == 'value2'
        assert config['abbreviations']['gh'] == 'https://github.com/{0}.git'

        # Test loading non-existent config
        non_existent_path = os.path.join(temp_dir, 'nonexistent.yml')
        with pytest.raises(ConfigDoesNotExistException):
            get_config(non_existent_path)

        # Test loading invalid YAML
        invalid_yaml_path = os.path.join(temp_dir, 'invalid.yml')
        with open(invalid_yaml_path, 'w', encoding='utf-8') as f:
            f.write('invalid: yaml: file')

        with pytest.raises(InvalidConfiguration):
            get_config(invalid_yaml_path)

        # Test loading YAML with non-dict top-level element
        non_dict_path = os.path.join(temp_dir, 'non_dict.yml')
        with open(non_dict_path, 'w', encoding='utf-8') as f:
            f.write('- item1\n- item2\n')

        with pytest.raises(InvalidConfiguration):
            get_config(non_dict_path)

    finally:
        # Clean up
        shutil.rmtree(temp_dir)


# LLM-generated content at query #9
#--------------------------

# Unit test for function get_config
def test_get_config():
    import tempfile
    import shutil

    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()

    # Create a temporary config file
    config_file = os.path.join(temp_dir, 'config.yml')
    with open(config_file, 'w') as f:
        f.write('cookiecutters_dir: /tmp/cookiecutters\n')
        f.write('replay_dir: /tmp/replay\n')
        f.write('default_context:\n')
        f.write('  key1: value1\n')
        f.write('  key2: value2\n')
        f.write('abbreviations:\n')
        f.write('  gh: https://github.com/{0}.git\n')

    # Test get_config
    config = get_config(config_file)
    assert config['cookiecutters_dir'] == '/tmp/cookiecutters'
    assert config['replay_dir'] == '/tmp/replay'
    assert config['default_context'] == {'key1': 'value1', 'key2': 'value2'}
    assert config['abbreviations'] == {'gh': 'https://github.com/{0}.git'}

    # Clean up
    shutil.rmtree(temp_dir)


# LLM-generated content at query #10
#--------------------------

# Unit test for function get_config
def test_get_config():
    """Test the get_config function."""
    import tempfile
    import shutil
    import pytest

    # Setup a temporary directory
    temp_dir = tempfile.mkdtemp()
    config_path = os.path.join(temp_dir, 'test_config.yml')

    # Test case 1: Valid YAML config
    valid_config = {
        'cookiecutters_dir': '/custom/cookiecutters',
        'replay_dir': '/custom/replay',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://custom.com/{0}.git'},
    }
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(valid_config, f)
    result = get_config(config_path)
    assert result['cookiecutters_dir'] == '/custom/cookiecutters'
    assert result['replay_dir'] == '/custom/replay'
    assert result['default_context'] == {'key': 'value'}
    assert result['abbreviations']['custom'] == 'https://custom.com/{0}.git'
    assert result['abbreviations']['gh'] == 'https://github.com/{0}.git'  # Built-in preserved

    # Test case 2: Invalid YAML config
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write('invalid: yaml: here')
    with pytest.raises(InvalidConfiguration):
        get_config(config_path)

    # Test case 3: Non-dict top-level YAML
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write('- item1\n- item2')
    with pytest.raises(InvalidConfiguration):
        get_config(config_path)

    # Test case 4: Non-existent config file
    with pytest.raises(ConfigDoesNotExistException):
        get_config(os.path.join(temp_dir, 'nonexistent.yml'))

    # Cleanup
    shutil.rmtree(temp_dir)


# LLM-generated content at query #11
#--------------------------

# Unit test for function get_config
def test_get_config():
    # Test case 1: Valid configuration file
    valid_config_path = "valid_config.yaml"
    with open(valid_config_path, "w") as f:
        yaml.dump({"cookiecutters_dir": "/custom/cookiecutters_dir"}, f)
    assert get_config(valid_config_path)["cookiecutters_dir"] == "/custom/cookiecutters_dir"
    os.remove(valid_config_path)

    # Test case 2: Invalid YAML file
    invalid_yaml_path = "invalid_yaml.yaml"
    with open(invalid_yaml_path, "w") as f:
        f.write("invalid: yaml: file")
    try:
        get_config(invalid_yaml_path)
        assert False, "Expected InvalidConfiguration exception"
    except InvalidConfiguration:
        pass
    os.remove(invalid_yaml_path)

    # Test case 3: Non-existent configuration file
    non_existent_path = "non_existent.yaml"
    try:
        get_config(non_existent_path)
        assert False, "Expected ConfigDoesNotExistException exception"
    except ConfigDoesNotExistException:
        pass

    # Test case 4: Configuration file with top-level non-dict element
    non_dict_path = "non_dict.yaml"
    with open(non_dict_path, "w") as f:
        f.write("not a dict")
    try:
        get_config(non_dict_path)
        assert False, "Expected InvalidConfiguration exception"
    except InvalidConfiguration:
        pass
    os.remove(non_dict_path)

    # Test case 5: Configuration file with environment variables
    env_var_config_path = "env_var_config.yaml"
    with open(env_var_config_path, "w") as f:
        yaml.dump({"cookiecutters_dir": "$HOME/custom_cookiecutters"}, f)
    assert get_config(env_var_config_path)["cookiecutters_dir"] == os.path.expanduser("~/custom_cookiecutters")
    os.remove(env_var_config_path)

    # Test case 6: Configuration file with nested dictionaries
    nested_dict_path = "nested_dict.yaml"
    with open(nested_dict_path, "w") as f:
        yaml.dump({"default_context": {"key1": "value1", "key2": "value2"}}, f)
    assert get_config(nested_dict_path)["default_context"] == {"key1": "value1", "key2": "value2"}
    os.remove(nested_dict_path)


# LLM-generated content at query #12
#--------------------------

# Unit test for function get_config
def test_get_config():
    # Test case 1: Valid config file
    config_file = "valid_config.yaml"
    with open(config_file, "w") as f:
        f.write("cookiecutters_dir: /custom/cookiecutters/\nreplay_dir: /custom/replay/")
    config = get_config(config_file)
    assert config["cookiecutters_dir"] == "/custom/cookiecutters/"
    assert config["replay_dir"] == "/custom/replay/"
    os.remove(config_file)

    # Test case 2: Non-existent config file
    try:
        get_config("nonexistent.yaml")
        assert False, "Expected ConfigDoesNotExistException"
    except ConfigDoesNotExistException:
        pass

    # Test case 3: Invalid YAML file
    invalid_yaml_file = "invalid.yaml"
    with open(invalid_yaml_file, "w") as f:
        f.write("invalid: yaml: file")
    try:
        get_config(invalid_yaml_file)
        assert False, "Expected InvalidConfiguration"
    except InvalidConfiguration:
        pass
    os.remove(invalid_yaml_file)

    # Test case 4: Non-dict top-level element
    invalid_top_level_file = "invalid_top_level.yaml"
    with open(invalid_top_level_file, "w") as f:
        f.write("- item1\n- item2")
    try:
        get_config(invalid_top_level_file)
        assert False, "Expected InvalidConfiguration"
    except InvalidConfiguration:
        pass
    os.remove(invalid_top_level_file)



# LLM-generated content at query #13
#--------------------------

# Unit test for function get_user_config
def test_get_user_config():
    # Test default_config=True
    assert get_user_config(default_config=True) == DEFAULT_CONFIG

    # Test default_config as dict
    custom_config = {'cookiecutters_dir': '/custom/path'}
    expected_config = DEFAULT_CONFIG.copy()
    expected_config.update(custom_config)
    assert get_user_config(default_config=custom_config) == expected_config

    # Test with custom config file path
    # Assuming a mock config file exists at '/mock/path' with content:
    # cookiecutters_dir: /mock/path
    # Replay_dir: /mock/replay
    mock_config = {'cookiecutters_dir': '/mock/path', 'replay_dir': '/mock/replay'}
    expected_config = DEFAULT_CONFIG.copy()
    expected_config.update(mock_config)
    assert get_user_config(config_file='/mock/path') == expected_config

    # Test with environment variable set
    os.environ['COOKIECUTTER_CONFIG'] = '/env/path'
    mock_config_env = {'cookiecutters_dir': '/env/path', 'replay_dir': '/env/replay'}
    expected_config_env = DEFAULT_CONFIG.copy()
    expected_config_env.update(mock_config_env)
    assert get_user_config() == expected_config_env
    del os.environ['COOKIECUTTER_CONFIG']

    # Test with default USER_CONFIG_PATH
    # Assuming USER_CONFIG_PATH exists with content:
    # cookiecutters_dir: ~/.cookiecutters/
    # replay_dir: ~/.cookiecutter_replay/
    assert get_user_config() == DEFAULT_CONFIG

    # Test with non-existent config file
    try:
        get_user_config(config_file='/non/existent/path')
    except ConfigDoesNotExistException:
        pass
    else:
        assert False, "Expected ConfigDoesNotExistException"

    # Test with invalid YAML file
    try:
        get_user_config(config_file='/invalid/yaml/path')
    except InvalidConfiguration:
        pass
    else:
        assert False, "Expected InvalidConfiguration"


# LLM-generated content at query #14
#--------------------------

# Unit test for function get_user_config
def test_get_user_config():
    # Test default config
    assert get_user_config(default_config=True) == DEFAULT_CONFIG

    # Test custom config file
    custom_config = {'cookiecutters_dir': '/custom/path'}
    assert get_user_config(default_config=custom_config) == merge_configs(DEFAULT_CONFIG, custom_config)

    # Test environment variable config
    os.environ['COOKIECUTTER_CONFIG'] = '/env/config/path'
    assert get_user_config() == get_config('/env/config/path')

    # Test default config file path
    del os.environ['COOKIECUTTER_CONFIG']
    if os.path.exists(USER_CONFIG_PATH):
        assert get_user_config() == get_config(USER_CONFIG_PATH)
    else:
        assert get_user_config() == DEFAULT_CONFIG

    # Test custom config file path
    assert get_user_config(config_file='/custom/config/path') == get_config('/custom/config/path')


# LLM-generated content at query #15
#--------------------------

# Unit test for function get_config
def test_get_config():
    # Test with a valid config file
    config_content = {
        'cookiecutters_dir': '/custom/cookiecutters/',
        'replay_dir': '/custom/replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://custom.com/{0}'},
    }
    with open('valid_config.yml', 'w', encoding='utf-8') as file:
        yaml.dump(config_content, file)
    config = get_config('valid_config.yml')
    assert config['cookiecutters_dir'] == '/custom/cookiecutters/'
    assert config['replay_dir'] == '/custom/replay/'
    assert config['default_context'] == {'key': 'value'}
    assert config['abbreviations'] == merge_configs(BUILTIN_ABBREVIATIONS, {'custom': 'https://custom.com/{0}'})
    os.remove('valid_config.yml')

    # Test with a non-existent config file
    try:
        get_config('non_existent_config.yml')
    except ConfigDoesNotExistException:
        pass
    else:
        assert False, "Expected ConfigDoesNotExistException"

    # Test with an invalid YAML file
    with open('invalid_config.yml', 'w', encoding='utf-8') as file:
        file.write("invalid yaml content")
    try:
        get_config('invalid_config.yml')
    except InvalidConfiguration:
        pass
    else:
        assert False, "Expected InvalidConfiguration"
    os.remove('invalid_config.yml')

    # Test with a YAML file that is not a dictionary
    with open('not_dict_config.yml', 'w', encoding='utf-8') as file:
        file.write("- item1\n- item2")
    try:
        get_config('not_dict_config.yml')
    except InvalidConfiguration:
        pass
    else:
        assert False, "Expected InvalidConfiguration"
    os.remove('not_dict_config.yml')



# LLM-generated content at query #16
#--------------------------

# Unit test for function get_user_config
def test_get_user_config():
    # Test default config
    assert get_user_config(default_config=True) == DEFAULT_CONFIG

    # Test custom config merge
    custom_config = {'cookiecutters_dir': '/custom/path'}
    merged = get_user_config(default_config=custom_config)
    assert merged['cookiecutters_dir'] == '/custom/path'
    assert merged['replay_dir'] == DEFAULT_CONFIG['replay_dir']

    # Test invalid config file raises exception
    try:
        get_user_config(config_file='/nonexistent/path')
        assert False, "Expected ConfigDoesNotExistException"
    except ConfigDoesNotExistException:
        pass

    # Test environment variable config
    os.environ['COOKIECUTTER_CONFIG'] = USER_CONFIG_PATH
    try:
        assert get_user_config() == get_config(USER_CONFIG_PATH)
    finally:
        del os.environ['COOKIECUTTER_CONFIG']


# LLM-generated content at query #17
#--------------------------

# Unit test for function get_config
def test_get_config():
    """Test the get_config function."""
    import tempfile
    from pathlib import Path

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a temporary config file
        config_file = Path(tmpdir) / 'config.yaml'
        config_file.write_text(
            'cookiecutters_dir: /tmp/cookiecutters\n'
            'replay_dir: /tmp/replay\n'
            'default_context:\n'
            '  foo: bar\n'
            'abbreviations:\n'
            '  gh: https://github.com/{0}.git\n'
        )

        # Test the function
        config = get_config(config_file)
        assert config['cookiecutters_dir'] == '/tmp/cookiecutters'
        assert config['replay_dir'] == '/tmp/replay'
        assert config['default_context'] == {'foo': 'bar'}
        assert config['abbreviations'] == {'gh': 'https://github.com/{0}.git'}

        # Test with a non-existent file
        try:
            get_config(Path(tmpdir) / 'nonexistent.yaml')
        except ConfigDoesNotExistException:
            pass
        else:
            assert False, "Expected ConfigDoesNotExistException"

        # Test with an invalid YAML file
        invalid_yaml_file = Path(tmpdir) / 'invalid.yaml'
        invalid_yaml_file.write_text('invalid: yaml: file')
        try:
            get_config(invalid_yaml_file)
        except InvalidConfiguration:
            pass
        else:
            assert False, "Expected InvalidConfiguration"

        # Test with a YAML file that is not a dict
        not_dict_yaml_file = Path(tmpdir) / 'not_dict.yaml'
        not_dict_yaml_file.write_text('not a dict')
        try:
            get_config(not_dict_yaml_file)
        except InvalidConfiguration:
            pass
        else:
            assert False, "Expected InvalidConfiguration"


# LLM-generated content at query #18
#--------------------------

# Unit test for function get_config
def test_get_config():
    import tempfile
    import pytest
    from pathlib import Path

    # Create a temporary config file
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as config_file:
        config_file.write('cookiecutters_dir: /tmp/cookiecutters\n')
        config_file.write('replay_dir: /tmp/replay\n')
        config_file.write('default_context:\n')
        config_file.write('  key1: value1\n')
        config_file.write('abbreviations:\n')
        config_file.write('  gh: https://github.com/{0}.git\n')
        config_file_path = Path(config_file.name)

    # Test valid YAML file
    config = get_config(config_file_path)
    assert isinstance(config, dict)
    assert config['cookiecutters_dir'] == '/tmp/cookiecutters'
    assert config['replay_dir'] == '/tmp/replay'
    assert config['default_context']['key1'] == 'value1'
    assert config['abbreviations']['gh'] == 'https://github.com/{0}.git'

    # Test non-existent file
    non_existent_file = Path('/nonexistent/path')
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test invalid YAML file
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as invalid_yaml_file:
        invalid_yaml_file.write('invalid yaml')
        invalid_yaml_file_path = Path(invalid_yaml_file.name)
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file_path)

    # Test top-level element is not a dict
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as invalid_top_level_file:
        invalid_top_level_file.write('- item1\n')
        invalid_top_level_file_path = Path(invalid_top_level_file.name)
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_top_level_file_path)

    # Clean up temporary files
    os.unlink(config_file_path)
    os.unlink(invalid_yaml_file_path)
    os.unlink(invalid_top_level_file_path)


# LLM-generated content at query #19
#--------------------------

# Unit test for function get_user_config
def test_get_user_config():
    # Test with default_config=True
    assert get_user_config(default_config=True) == DEFAULT_CONFIG

    # Test with default_config as a dict
    custom_config = {'cookiecutters_dir': '/custom/path'}
    expected_config = DEFAULT_CONFIG.copy()
    expected_config.update(custom_config)
    assert get_user_config(default_config=custom_config) == expected_config

    # Test with a custom config file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
        temp_file.write(yaml.dump(custom_config))
        temp_file_path = temp_file.name
    assert get_user_config(config_file=temp_file_path) == expected_config

    # Test with environment variable
    import os
    os.environ['COOKIECUTTER_CONFIG'] = temp_file_path
    assert get_user_config() == expected_config

    # Clean up
    os.unlink(temp_file_path)
    del os.environ['COOKIECUTTER_CONFIG']

    # Test with default config file path
    with open(USER_CONFIG_PATH, 'w') as f:
        f.write(yaml.dump(custom_config))
    assert get_user_config() == expected_config
    os.unlink(USER_CONFIG_PATH)

    # Test with no config file and no environment variable
    assert get_user_config() == DEFAULT_CONFIG


# LLM-generated content at query #20
#--------------------------

# Unit test for function get_config
def test_get_config():
    # Test with a valid config file
    config_file = 'tests/test-config.yaml'
    config = get_config(config_file)
    assert config['cookiecutters_dir'] == '/tmp/custom_cookiecutters'
    assert config['replay_dir'] == '/tmp/custom_replay'
    assert config['default_context']['full_name'] == 'Your Name'
    assert config['abbreviations']['gh'] == 'https://github.com/{0}.git'

    # Test with a config file that does not exist
    config_file = 'tests/nonexistent-config.yaml'
    try:
        get_config(config_file)
    except ConfigDoesNotExistException:
        pass
    else:
        assert False, "Expected ConfigDoesNotExistException"

    # Test with a config file that has invalid YAML
    config_file = 'tests/invalid-config.yaml'
    try:
        get_config(config_file)
    except InvalidConfiguration:
        pass
    else:
        assert False, "Expected InvalidConfiguration"

    # Test with a config file that has invalid top-level element
    config_file = 'tests/invalid-top-level-config.yaml'
    try:
        get_config(config_file)
    except InvalidConfiguration:
        pass
    else:
        assert False, "Expected InvalidConfiguration"


# LLM-generated content at query #21
#--------------------------

# Unit test for function get_config
def test_get_config():
    import tempfile
    from pathlib import Path

    # Create a temporary YAML config file
    config_content = """
    cookiecutters_dir: ~/custom_cookiecutters/
    replay_dir: ~/custom_replay/
    default_context:
        key: value
    abbreviations:
        custom: https://custom.com/{0}.git
    """
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
        temp_file.write(config_content)
        temp_file_path = temp_file.name

    try:
        # Test loading the config
        config = get_config(temp_file_path)
        assert config['cookiecutters_dir'] == os.path.expanduser('~/custom_cookiecutters/')
        assert config['replay_dir'] == os.path.expanduser('~/custom_replay/')
        assert config['default_context']['key'] == 'value'
        assert config['abbreviations']['custom'] == 'https://custom.com/{0}.git'
        assert config['abbreviations']['gh'] == 'https://github.com/{0}.git'  # Check default abbreviation remains

        # Test invalid YAML
        invalid_config_content = "invalid: yaml: content"
        with open(temp_file_path, 'w') as temp_file:
            temp_file.write(invalid_config_content)
        try:
            get_config(temp_file_path)
            assert False, "Expected InvalidConfiguration exception"
        except InvalidConfiguration:
            pass

        # Test non-dict YAML
        non_dict_config_content = "not_a_dict"
        with open(temp_file_path, 'w') as temp_file:
            temp_file.write(non_dict_config_content)
        try:
            get_config(temp_file_path)
            assert False, "Expected InvalidConfiguration exception"
        except InvalidConfiguration:
            pass

    finally:
        # Clean up the temporary file
        Path(temp_file_path).unlink()



# LLM-generated content at query #22
#--------------------------

# Unit test for function get_config
def test_get_config():
    # Setup: Create a temporary config file
    import tempfile
    import os
    from pathlib import Path

    # Test case 1: Valid config file
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp_file:
        tmp_file.write("cookiecutters_dir: /tmp/cookiecutters\n")
        tmp_file.write("replay_dir: /tmp/replay\n")
        tmp_file.write("default_context:\n")
        tmp_file.write("  key1: value1\n")
        tmp_file.write("abbreviations:\n")
        tmp_file.write("  gh: https://github.com/{0}.git\n")
        tmp_file_path = tmp_file.name

    # Execute
    config = get_config(Path(tmp_file_path))

    # Verify
    assert config['cookiecutters_dir'] == '/tmp/cookiecutters'
    assert config['replay_dir'] == '/tmp/replay'
    assert config['default_context']['key1'] == 'value1'
    assert config['abbreviations']['gh'] == 'https://github.com/{0}.git'

    # Cleanup
    os.remove(tmp_file_path)

    # Test case 2: Config file does not exist
    try:
        get_config(Path('/nonexistent/path'))
    except ConfigDoesNotExistException:
        pass
    else:
        assert False, "Expected ConfigDoesNotExistException"

    # Test case 3: Invalid YAML file
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp_file:
        tmp_file.write("invalid yaml content")
        tmp_file_path = tmp_file.name

    try:
        get_config(Path(tmp_file_path))
    except InvalidConfiguration:
        pass
    else:
        assert False, "Expected InvalidConfiguration"

    # Cleanup
    os.remove(tmp_file_path)

    # Test case 4: Top-level element is not a dict
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp_file:
        tmp_file.write("- item1\n")
        tmp_file.write("- item2\n")
        tmp_file_path = tmp_file.name

    try:
        get_config(Path(tmp_file_path))
    except InvalidConfiguration:
        pass
    else:
        assert False, "Expected InvalidConfiguration"

    # Cleanup
    os.remove(tmp_file_path)


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

# Unit test for function get_config
def test_get_config():
    # Test with valid config file
    config = get_config('tests/test_config.yaml')
    assert config['cookiecutters_dir'] == '/tmp/test_cookiecutters'
    assert config['replay_dir'] == '/tmp/test_replay'
    assert config['default_context']['project_name'] == 'Test Project'
    assert config['abbreviations']['gh'] == 'https://github.com/{0}.git'

    # Test with invalid config file
    try:
        get_config('tests/invalid_config.yaml')
        assert False
    except InvalidConfiguration:
        assert True

    # Test with non-existent config file
    try:
        get_config('tests/non_existent_config.yaml')
        assert False
    except ConfigDoesNotExistException:
        assert True


# LLM-generated content at query #2
#--------------------------

# Unit test for function get_config
def test_get_config():
    import tempfile
    import pytest

    # Test with a valid config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml') as tmp:
        tmp.write('cookiecutters_dir: /tmp/cookies\nreplay_dir: /tmp/replay')
        tmp.flush()
        config = get_config(tmp.name)
        assert config['cookiecutters_dir'] == '/tmp/cookies'
        assert config['replay_dir'] == '/tmp/replay'

    # Test with a non-existent config file
    with pytest.raises(ConfigDoesNotExistException):
        get_config('/nonexistent/path')

    # Test with an invalid YAML file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml') as tmp:
        tmp.write('invalid: yaml: here')
        tmp.flush()
        with pytest.raises(InvalidConfiguration):
            get_config(tmp.name)

    # Test with a YAML file that's not a dict
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml') as tmp:
        tmp.write('- list\n- items')
        tmp.flush()
        with pytest.raises(InvalidConfiguration):
            get_config(tmp.name)


# LLM-generated content at query #3
#--------------------------

# Unit test for function get_config
def test_get_config():
    # Create a temporary config file
    import tempfile
    import json

    # Test case 1: Valid config file
    config_content = {
        'cookiecutters_dir': '/tmp/cookiecutters',
        'replay_dir': '/tmp/replay',
        'default_context': {'key': 'value'},
        'abbreviations': {'test': 'https://example.com/{0}.git'},
    }
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
        yaml.dump(config_content, temp_file)
        temp_file_path = temp_file.name

    try:
        config = get_config(temp_file_path)
        assert config['cookiecutters_dir'] == '/tmp/cookiecutters'
        assert config['replay_dir'] == '/tmp/replay'
        assert config['default_context'] == {'key': 'value'}
        assert config['abbreviations']['test'] == 'https://example.com/{0}.git'
    finally:
        os.remove(temp_file_path)

    # Test case 2: Config file does not exist
    try:
        get_config('/path/to/nonexistent/file')
    except ConfigDoesNotExistException:
        pass
    else:
        assert False, "Expected ConfigDoesNotExistException"

    # Test case 3: Invalid YAML file
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
        temp_file.write('invalid yaml content')
        temp_file_path = temp_file.name

    try:
        get_config(temp_file_path)
    except InvalidConfiguration:
        pass
    else:
        assert False, "Expected InvalidConfiguration"
    finally:
        os.remove(temp_file_path)

    # Test case 4: Top-level element is not a dict
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
        yaml.dump([1, 2, 3], temp_file)
        temp_file_path = temp_file.name

    try:
        get_config(temp_file_path)
    except InvalidConfiguration:
        pass
    else:
        assert False, "Expected InvalidConfiguration"
    finally:
        os.remove(temp_file_path)



# LLM-generated content at query #4
#--------------------------

# Unit test for function get_config
def test_get_config():
    import tempfile
    import shutil

    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()

    try:
        # Create a config file
        config_file_path = os.path.join(temp_dir, 'test_config.yml')
        config_data = {
            'cookiecutters_dir': '~/custom_cookiecutters',
            'replay_dir': '~/custom_replay',
            'default_context': {'key1': 'value1'},
            'abbreviations': {'custom': 'https://custom.com/{0}'}
        }
        with open(config_file_path, 'w', encoding='utf-8') as file:
            yaml.dump(config_data, file)

        # Call get_config
        result = get_config(config_file_path)

        # Verify the result
        assert result['cookiecutters_dir'] == os.path.expanduser('~/custom_cookiecutters')
        assert result['replay_dir'] == os.path.expanduser('~/custom_replay')
        assert result['default_context'] == {'key1': 'value1'}
        assert result['abbreviations'] == merge_configs(BUILTIN_ABBREVIATIONS, {'custom': 'https://custom.com/{0}'})

    finally:
        # Clean up the temporary directory
        shutil.rmtree(temp_dir)


# LLM-generated content at query #5
#--------------------------

# Unit test for function get_config
def test_get_config():
    import tempfile
    import pytest

    # Test with a valid config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml') as tmp:
        tmp.write('cookiecutters_dir: /tmp/cookies\nreplay_dir: /tmp/replay')
        tmp.flush()
        config = get_config(tmp.name)
        assert config['cookiecutters_dir'] == '/tmp/cookies'
        assert config['replay_dir'] == '/tmp/replay'

    # Test with an invalid YAML file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml') as tmp:
        tmp.write('invalid: yaml: file')
        tmp.flush()
        with pytest.raises(InvalidConfiguration):
            get_config(tmp.name)

    # Test with a non-existent file
    with pytest.raises(ConfigDoesNotExistException):
        get_config('/nonexistent/path')

    # Test with a file that's not a dict
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml') as tmp:
        tmp.write('- list\n- items')
        tmp.flush()
        with pytest.raises(InvalidConfiguration):
            get_config(tmp.name)


# LLM-generated content at query #6
#--------------------------

# Unit test for function get_config
def test_get_config():
    """Test the get_config function."""
    import tempfile
    import shutil
    import pytest

    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    try:
        # Create a config file with valid YAML
        valid_config_path = os.path.join(temp_dir, 'valid_config.yaml')
        with open(valid_config_path, 'w', encoding='utf-8') as f:
            f.write('cookiecutters_dir: /tmp/custom_cookiecutters\n')
            f.write('replay_dir: /tmp/custom_replay\n')
            f.write('default_context:\n')
            f.write('  key1: value1\n')
            f.write('abbreviations:\n')
            f.write('  gh: "https://github.com/{0}.git"\n')

        # Test loading valid config
        config = get_config(valid_config_path)
        assert config['cookiecutters_dir'] == '/tmp/custom_cookiecutters'
        assert config['replay_dir'] == '/tmp/custom_replay'
        assert config['default_context']['key1'] == 'value1'
        assert config['abbreviations']['gh'] == 'https://github.com/{0}.git'

        # Create a config file with invalid YAML
        invalid_yaml_path = os.path.join(temp_dir, 'invalid_yaml.yaml')
        with open(invalid_yaml_path, 'w', encoding='utf-8') as f:
            f.write('invalid: yaml: here')

        # Test loading invalid YAML
        with pytest.raises(InvalidConfiguration):
            get_config(invalid_yaml_path)

        # Create a config file with non-dict top-level element
        non_dict_path = os.path.join(temp_dir, 'non_dict.yaml')
        with open(non_dict_path, 'w', encoding='utf-8') as f:
            f.write('- item1\n- item2')

        # Test loading non-dict YAML
        with pytest.raises(InvalidConfiguration):
            get_config(non_dict_path)

        # Test non-existent config file
        non_existent_path = os.path.join(temp_dir, 'nonexistent.yaml')
        with pytest.raises(ConfigDoesNotExistException):
            get_config(non_existent_path)

    finally:
        # Clean up
        shutil.rmtree(temp_dir)


# LLM-generated content at query #7
#--------------------------

# Unit test for function get_config
def test_get_config():
    # Write a temporary config file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as config_file:
        config_file.write('cookiecutters_dir: /tmp/cookiecutters\n')
        config_file.write('replay_dir: /tmp/replay\n')
        config_file.write('default_context:\n')
        config_file.write('  key1: value1\n')
        config_file.write('abbreviations:\n')
        config_file.write('  gh: https://github.com/{0}.git\n')
        config_file_path = config_file.name

    # Test the function
    config = get_config(config_file_path)
    assert config['cookiecutters_dir'] == '/tmp/cookiecutters'
    assert config['replay_dir'] == '/tmp/replay'
    assert config['default_context'] == {'key1': 'value1'}
    assert config['abbreviations'] == {'gh': 'https://github.com/{0}.git'}

    # Clean up
    os.remove(config_file_path)


# LLM-generated content at query #8
#--------------------------

# Unit test for function get_config
def test_get_config():
    """Test the get_config function."""
    import tempfile
    import shutil
    from pathlib import Path

    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    config_path = Path(temp_dir) / 'test_config.yml'

    try:
        # Test with valid YAML
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write('cookiecutters_dir: /custom/cookiecutters\nreplay_dir: /custom/replay')
        config = get_config(config_path)
        assert config['cookiecutters_dir'] == '/custom/cookiecutters'
        assert config['replay_dir'] == '/custom/replay'

        # Test with invalid YAML
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write('invalid: yaml: here')
        try:
            get_config(config_path)
            assert False, "Should have raised InvalidConfiguration"
        except InvalidConfiguration:
            pass

        # Test with non-dict YAML
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write('- list\n- item')
        try:
            get_config(config_path)
            assert False, "Should have raised InvalidConfiguration"
        except InvalidConfiguration:
            pass

        # Test with non-existent file
        non_existent = Path(temp_dir) / 'nonexistent.yml'
        try:
            get_config(non_existent)
            assert False, "Should have raised ConfigDoesNotExistException"
        except ConfigDoesNotExistException:
            pass

    finally:
        shutil.rmtree(temp_dir)


# LLM-generated content at query #9
#--------------------------

# Unit test for function get_config
def test_get_config():
    import tempfile
    import pathlib

    # Test with a valid config file
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write('cookiecutters_dir: /tmp/cookiecutters\n')
        f.write('replay_dir: /tmp/cookiecutter_replay\n')
        f.write('default_context:\n')
        f.write('  key1: value1\n')
        f.write('abbreviations:\n')
        f.write('  gh: https://github.com/{0}.git\n')
        f.write('  gl: https://gitlab.com/{0}.git\n')
        f.write('  bb: https://bitbucket.org/{0}\n')
        f.close()
        config = get_config(pathlib.Path(f.name))
        assert config['cookiecutters_dir'] == '/tmp/cookiecutters'
        assert config['replay_dir'] == '/tmp/cookiecutter_replay'
        assert config['default_context']['key1'] == 'value1'
        assert config['abbreviations']['gh'] == 'https://github.com/{0}.git'
        assert config['abbreviations']['gl'] == 'https://gitlab.com/{0}.git'
        assert config['abbreviations']['bb'] == 'https://bitbucket.org/{0}'
        os.unlink(f.name)

    # Test with an invalid config file
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write('cookiecutters_dir: /tmp/cookiecutters\n')
        f.write('replay_dir: /tmp/cookiecutter_replay\n')
        f.write('default_context:\n')
        f.write('  key1: value1\n')
        f.write('abbreviations:\n')
        f.write('  gh: https://github.com/{0}.git\n')
        f.write('  gl: https://gitlab.com/{0}.git\n')
        f.write('  bb: https://bitbucket.org/{0}\n')
        f.write('invalid_key: invalid_value\n')
        f.close()
        config = get_config(pathlib.Path(f.name))
        assert config['cookiecutters_dir'] == '/tmp/cookiecutters'
        assert config['replay_dir'] == '/tmp/cookiecutter_replay'
        assert config['default_context']['key1'] == 'value1'
        assert config['abbreviations']['gh'] == 'https://github.com/{0}.git'
        assert config['abbreviations']['gl'] == 'https://gitlab.com/{0}.git'
        assert config['abbreviations']['bb'] == 'https://bitbucket.org/{0}'
        assert config['invalid_key'] == 'invalid_value'
        os.unlink(f.name)

    # Test with a non-existent config file
    try:
        config = get_config(pathlib.Path('/tmp/non_existent_file'))
    except ConfigDoesNotExistException:
        pass
    else:
        assert False, 'Expected ConfigDoesNotExistException'


# LLM-generated content at query #10
#--------------------------

# Unit test for function get_config
def test_get_config():
    # Create a temporary config file for testing
    import tempfile
    import shutil
    temp_dir = tempfile.mkdtemp()
    temp_config_path = os.path.join(temp_dir, 'test_config.yml')

    # Test case: Valid config file
    test_config = {
        'cookiecutters_dir': '/tmp/cookiecutters',
        'replay_dir': '/tmp/replay',
        'default_context': {'key': 'value'},
        'abbreviations': {'gh': 'https://github.com/{0}.git'},
    }
    with open(temp_config_path, 'w', encoding='utf-8') as f:
        yaml.dump(test_config, f)
    config = get_config(temp_config_path)
    assert config['cookiecutters_dir'] == '/tmp/cookiecutters'
    assert config['replay_dir'] == '/tmp/replay'
    assert config['default_context'] == {'key': 'value'}
    assert config['abbreviations'] == {'gh': 'https://github.com/{0}.git'}

    # Test case: Config file does not exist
    try:
        get_config('/invalid/path/to/config.yml')
    except ConfigDoesNotExistException:
        pass
    else:
        assert False, "Expected ConfigDoesNotExistException"

    # Test case: Invalid YAML file
    with open(temp_config_path, 'w', encoding='utf-8') as f:
        f.write('invalid yaml')
    try:
        get_config(temp_config_path)
    except InvalidConfiguration:
        pass
    else:
        assert False, "Expected InvalidConfiguration"

    # Clean up
    shutil.rmtree(temp_dir)


# LLM-generated content at query #11
#--------------------------

# Unit test for function get_user_config
def test_get_user_config():
    # Test with default_config=True
    assert get_user_config(default_config=True) == DEFAULT_CONFIG

    # Test with default_config as a dict
    custom_config = {'cookiecutters_dir': '/custom/path'}
    expected = merge_configs(DEFAULT_CONFIG, custom_config)
    assert get_user_config(default_config=custom_config) == expected

    # Test with custom config file
    # Mocking file existence and content is needed here
    # For simplicity, assume the file exists and contains valid YAML
    custom_file = '/fake/config.yaml'
    custom_content = {'replay_dir': '/custom/replay'}
    expected = merge_configs(DEFAULT_CONFIG, custom_content)
    assert get_user_config(config_file=custom_file) == expected

    # Test with environment variable
    os.environ['COOKIECUTTER_CONFIG'] = '/env/config.yaml'
    env_content = {'default_context': {'key': 'value'}}
    expected = merge_configs(DEFAULT_CONFIG, env_content)
    assert get_user_config() == expected

    # Test with default user config file
    # Mocking file existence and content is needed here
    # For simplicity, assume the file exists and contains valid YAML
    user_content = {'abbreviations': {'custom': 'https://custom.com/{0}'}}
    expected = merge_configs(DEFAULT_CONFIG, user_content)
    assert get_user_config() == expected

    # Test with no config files or environment variables
    del os.environ['COOKIECUTTER_CONFIG']
    assert get_user_config() == DEFAULT_CONFIG


# LLM-generated content at query #12
#--------------------------

# Unit test for function get_config
def test_get_config():
    import tempfile
    from pathlib import Path

    # Test case 1: Valid YAML file
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp_file:
        tmp_file.write("cookiecutters_dir: /tmp/cookiecutters\n")
        tmp_file.write("replay_dir: /tmp/replay\n")
        tmp_file.write("default_context:\n")
        tmp_file.write("  key: value\n")
        tmp_file.write("abbreviations:\n")
        tmp_file.write("  gh: https://github.com/{0}.git\n")
        tmp_file_path = Path(tmp_file.name)

    config = get_config(tmp_file_path)
    assert config['cookiecutters_dir'] == '/tmp/cookiecutters'
    assert config['replay_dir'] == '/tmp/replay'
    assert config['default_context'] == collections.OrderedDict([('key', 'value')])
    assert config['abbreviations'] == {'gh': 'https://github.com/{0}.git'}

    # Test case 2: Invalid YAML file
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp_file:
        tmp_file.write("invalid: yaml: file\n")
        tmp_file_path = Path(tmp_file.name)

    try:
        get_config(tmp_file_path)
    except InvalidConfiguration:
        pass
    else:
        assert False, "Expected InvalidConfiguration exception"

    # Test case 3: Non-existent file
    non_existent_path = Path('/tmp/non_existent_file')
    try:
        get_config(non_existent_path)
    except ConfigDoesNotExistException:
        pass
    else:
        assert False, "Expected ConfigDoesNotExistException exception"

    # Test case 4: Invalid top-level element
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp_file:
        tmp_file.write("- item1\n")
        tmp_file.write("- item2\n")
        tmp_file_path = Path(tmp_file.name)

    try:
        get_config(tmp_file_path)
    except InvalidConfiguration:
        pass
    else:
        assert False, "Expected InvalidConfiguration exception"

    # Clean up temporary files
    os.unlink(tmp_file_path)



# LLM-generated content at query #13
#--------------------------

# Unit test for function get_config
def test_get_config():
    import tempfile
    import pytest

    # Create a temporary config file
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp:
        tmp.write('cookiecutters_dir: /custom/path\n')
        tmp.write('replay_dir: /custom/replay\n')
        tmp.write('default_context:\n')
        tmp.write('  key1: value1\n')
        tmp.write('abbreviations:\n')
        tmp.write('  gh: https://custom.github.com/{0}.git\n')
        tmp_path = tmp.name

    # Test valid config
    config = get_config(tmp_path)
    assert config['cookiecutters_dir'] == '/custom/path'
    assert config['replay_dir'] == '/custom/replay'
    assert config['default_context']['key1'] == 'value1'
    assert config['abbreviations']['gh'] == 'https://custom.github.com/{0}.git'

    # Test non-existent file
    with pytest.raises(ConfigDoesNotExistException):
        get_config('/nonexistent/path')

    # Test invalid YAML
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp:
        tmp.write('invalid: yaml: file')
        tmp_path = tmp.name
    with pytest.raises(InvalidConfiguration):
        get_config(tmp_path)

    # Test non-dict YAML
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp:
        tmp.write('- list\n- item')
        tmp_path = tmp.name
    with pytest.raises(InvalidConfiguration):
        get_config(tmp_path)

    # Clean up
    os.unlink(tmp_path)


# LLM-generated content at query #14
#--------------------------

# Unit test for function get_user_config
def test_get_user_config():
    # Test default config
    assert get_user_config(default_config=True) == DEFAULT_CONFIG

    # Test custom default config
    custom_default = {'cookiecutters_dir': '/custom/path'}
    expected = merge_configs(DEFAULT_CONFIG, custom_default)
    assert get_user_config(default_config=custom_default) == expected

    # Test loading from custom config file (mocked)
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp:
        tmp.write(yaml.dump({'cookiecutters_dir': '/tmp/custom'}))
        tmp_path = tmp.name

    try:
        result = get_user_config(config_file=tmp_path)
        assert result['cookiecutters_dir'] == '/tmp/custom'
    finally:
        os.unlink(tmp_path)

    # Test environment variable config (mocked)
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp:
        tmp.write(yaml.dump({'cookiecutters_dir': '/tmp/env'}))
        tmp_path = tmp.name

    try:
        import os
        os.environ['COOKIECUTTER_CONFIG'] = tmp_path
        result = get_user_config()
        assert result['cookiecutters_dir'] == '/tmp/env'
    finally:
        os.unlink(tmp_path)
        del os.environ['COOKIECUTTER_CONFIG']

    # Test invalid config file
    import pytest
    with pytest.raises(ConfigDoesNotExistException):
        get_user_config(config_file='/nonexistent/path')

    # Test invalid YAML
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp:
        tmp.write('invalid: yaml: here')
        tmp_path = tmp.name

    try:
        with pytest.raises(InvalidConfiguration):
            get_user_config(config_file=tmp_path)
    finally:
        os.unlink(tmp_path)


# LLM-generated content at query #15
#--------------------------

# Unit test for function get_config
def test_get_config():
    import tempfile

    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
        temp_file.write('cookiecutters_dir: /tmp/cookiecutters\n')
        temp_file.write('replay_dir: /tmp/cookiecutter_replay\n')
        temp_file_name = temp_file.name

    # Test with the temporary file
    config = get_config(temp_file_name)
    assert config['cookiecutters_dir'] == '/tmp/cookiecutters'
    assert config['replay_dir'] == '/tmp/cookiecutter_replay'

    # Clean up the temporary file
    os.unlink(temp_file_name)

    # Test with a non-existent file
    try:
        get_config('/non/existent/path')
    except ConfigDoesNotExistException:
        pass
    else:
        raise AssertionError("Expected ConfigDoesNotExistException")

    # Test with an invalid YAML file
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
        temp_file.write('invalid yaml')
        temp_file_name = temp_file.name

    try:
        get_config(temp_file_name)
    except InvalidConfiguration:
        pass
    else:
        raise AssertionError("Expected InvalidConfiguration")

    # Clean up the temporary file
    os.unlink(temp_file_name)



# LLM-generated content at query #16
#--------------------------

# Unit test for function get_config
def test_get_config():
    """Test the get_config function."""
    import tempfile
    import shutil
    from pathlib import Path

    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    try:
        # Create a config file
        config_path = Path(temp_dir) / 'test_config.yml'
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write('cookiecutters_dir: /tmp/cookiecutters\n')
            f.write('replay_dir: /tmp/replay\n')
            f.write('default_context:\n')
            f.write('  key1: value1\n')
            f.write('  key2: value2\n')
            f.write('abbreviations:\n')
            f.write('  gh: https://github.com/{0}.git\n')

        # Test the config file
        config = get_config(config_path)
        assert config['cookiecutters_dir'] == '/tmp/cookiecutters'
        assert config['replay_dir'] == '/tmp/replay'
        assert config['default_context']['key1'] == 'value1'
        assert config['default_context']['key2'] == 'value2'
        assert config['abbreviations']['gh'] == 'https://github.com/{0}.git'

        # Test non-existent config file
        try:
            get_config(Path(temp_dir) / 'nonexistent.yml')
            assert False, "Expected ConfigDoesNotExistException"
        except ConfigDoesNotExistException:
            pass

        # Test invalid YAML
        invalid_config_path = Path(temp_dir) / 'invalid_config.yml'
        with open(invalid_config_path, 'w', encoding='utf-8') as f:
            f.write('invalid: yaml: file')

        try:
            get_config(invalid_config_path)
            assert False, "Expected InvalidConfiguration"
        except InvalidConfiguration:
            pass

        # Test non-dict YAML
        invalid_config_path = Path(temp_dir) / 'invalid_config2.yml'
        with open(invalid_config_path, 'w', encoding='utf-8') as f:
            f.write('- item1\n- item2\n')

        try:
            get_config(invalid_config_path)
            assert False, "Expected InvalidConfiguration"
        except InvalidConfiguration:
            pass

    finally:
        # Clean up
        shutil.rmtree(temp_dir)


# LLM-generated content at query #17
#--------------------------

# Unit test for function get_config
def test_get_config():
    """Test the get_config function."""
    import tempfile
    import pytest

    # Test with a valid config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml') as tmp:
        tmp.write('cookiecutters_dir: /custom/path\n')
        tmp.write('replay_dir: /another/path\n')
        tmp.write('default_context:\n  key: value\n')
        tmp.write('abbreviations:\n  gh: custom_gh_url\n')
        tmp.flush()

        config = get_config(tmp.name)
        assert config['cookiecutters_dir'] == '/custom/path'
        assert config['replay_dir'] == '/another/path'
        assert config['default_context']['key'] == 'value'
        assert config['abbreviations']['gh'] == 'custom_gh_url'

    # Test with a non-existent file
    with pytest.raises(ConfigDoesNotExistException):
        get_config('/nonexistent/path')

    # Test with invalid YAML
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml') as tmp:
        tmp.write('invalid: yaml: here')
        tmp.flush()
        with pytest.raises(InvalidConfiguration):
            get_config(tmp.name)

    # Test with non-dict YAML
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml') as tmp:
        tmp.write('- list\n- item')
        tmp.flush()
        with pytest.raises(InvalidConfiguration):
            get_config(tmp.name)


# LLM-generated content at query #18
#--------------------------

# Unit test for function get_config
def test_get_config():
    """Test the get_config function."""
    # Test with a valid config file
    valid_config = {
        'cookiecutters_dir': '/custom/cookiecutters',
        'replay_dir': '/custom/replay',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://custom.com/{0}.git'},
    }
    with open('valid_config.yml', 'w') as f:
        yaml.dump(valid_config, f)
    assert get_config('valid_config.yml') == {
        'cookiecutters_dir': '/custom/cookiecutters',
        'replay_dir': '/custom/replay',
        'default_context': {'key': 'value'},
        'abbreviations': {
            'gh': 'https://github.com/{0}.git',
            'gl': 'https://gitlab.com/{0}.git',
            'bb': 'https://bitbucket.org/{0}',
            'custom': 'https://custom.com/{0}.git',
        },
    }

    # Test with an invalid config file
    with open('invalid_config.yml', 'w') as f:
        f.write('invalid: yaml: file')
    try:
        get_config('invalid_config.yml')
        assert False, "Should have raised InvalidConfiguration"
    except InvalidConfiguration:
        pass

    # Test with a non-existent config file
    try:
        get_config('nonexistent_config.yml')
        assert False, "Should have raised ConfigDoesNotExistException"
    except ConfigDoesNotExistException:
        pass

    # Clean up
    os.remove('valid_config.yml')
    os.remove('invalid_config.yml')


# LLM-generated content at query #19
#--------------------------

# Unit test for function get_config
def test_get_config():
    import tempfile

    # Create a temporary YAML config file
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
        config_content = """
        cookiecutters_dir: /custom/cookiecutters/
        replay_dir: /custom/cookiecutter_replay/
        default_context:
            project_name: Test Project
        abbreviations:
            gh: https://github.com/{0}.git
        """
        temp_file.write(config_content)
        temp_file_path = temp_file.name

    # Test normal config loading
    config = get_config(temp_file_path)
    assert config['cookiecutters_dir'] == '/custom/cookiecutters/'
    assert config['replay_dir'] == '/custom/cookiecutter_replay/'
    assert config['default_context']['project_name'] == 'Test Project'
    assert config['abbreviations']['gh'] == 'https://github.com/{0}.git'

    # Test config file does not exist
    try:
        get_config('/nonexistent/path')
    except ConfigDoesNotExistException:
        pass
    else:
        raise AssertionError("Expected ConfigDoesNotExistException")

    # Test invalid YAML file
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
        invalid_yaml_content = "invalid: yaml: content"
        temp_file.write(invalid_yaml_content)
        invalid_yaml_path = temp_file.name

    try:
        get_config(invalid_yaml_path)
    except InvalidConfiguration:
        pass
    else:
        raise AssertionError("Expected InvalidConfiguration")

    # Test YAML file with top-level element not being a dict
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
        invalid_top_level_content = "invalid_top_level"
        temp_file.write(invalid_top_level_content)
        invalid_top_level_path = temp_file.name

    try:
        get_config(invalid_top_level_path)
    except InvalidConfiguration:
        pass
    else:
        raise AssertionError("Expected InvalidConfiguration")

    # Clean up temporary files
    os.unlink(temp_file_path)
    os.unlink(invalid_yaml_path)
    os.unlink(invalid_top_level_path)



# LLM-generated content at query #20
#--------------------------

# Unit test for function get_config
def test_get_config():
    """Test the get_config function."""
    import tempfile
    import pytest

    # Create a temporary config file
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_config:
        temp_config.write('cookiecutters_dir: /custom/cookiecutters\n')
        temp_config.write('replay_dir: /custom/replay\n')
        temp_config_path = temp_config.name

    # Test valid config
    config = get_config(temp_config_path)
    assert config['cookiecutters_dir'] == '/custom/cookiecutters'
    assert config['replay_dir'] == '/custom/replay'

    # Test invalid YAML
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as bad_config:
        bad_config.write('invalid: yaml: here')
        bad_config_path = bad_config.name

    with pytest.raises(InvalidConfiguration):
        get_config(bad_config_path)

    # Test non-existent file
    with pytest.raises(ConfigDoesNotExistException):
        get_config('/nonexistent/path')

    # Clean up
    os.unlink(temp_config_path)
    os.unlink(bad_config_path)


# LLM-generated content at query #21
#--------------------------

# Unit test for function get_config
def test_get_config():
    # Create a temporary config file
    import tempfile
    import shutil
    import pytest
    from pathlib import Path

    temp_dir = tempfile.mkdtemp()
    config_path = Path(temp_dir) / 'test_config.yaml'
    config_data = {
        'cookiecutters_dir': '~/custom_cookiecutters',
        'replay_dir': '~/custom_replay',
    }
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config_data, f)

    # Test valid config
    config = get_config(config_path)
    assert config['cookiecutters_dir'] == os.path.expanduser('~/custom_cookiecutters')
    assert config['replay_dir'] == os.path.expanduser('~/custom_replay')

    # Test invalid config file
    invalid_config_path = Path(temp_dir) / 'invalid_config.yaml'
    with open(invalid_config_path, 'w', encoding='utf-8') as f:
        f.write('invalid: yaml: file')

    with pytest.raises(InvalidConfiguration):
        get_config(invalid_config_path)

    # Test non-existent config file
    non_existent_config_path = Path(temp_dir) / 'non_existent.yaml'
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_config_path)

    # Cleanup
    shutil.rmtree(temp_dir)



# LLM-generated content at query #22
#--------------------------

# Unit test for function get_config
def test_get_config():
    """Test the get_config function."""
    import tempfile
    import shutil
    import pytest

    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    test_config_path = os.path.join(temp_dir, 'test_config.yml')

    # Test case 1: Valid YAML config
    with open(test_config_path, 'w', encoding='utf-8') as f:
        f.write('cookiecutters_dir: /custom/cookiecutters\nreplay_dir: /custom/replay')
    config = get_config(test_config_path)
    assert config['cookiecutters_dir'] == '/custom/cookiecutters'
    assert config['replay_dir'] == '/custom/replay'

    # Test case 2: Invalid YAML config
    with open(test_config_path, 'w', encoding='utf-8') as f:
        f.write('invalid: yaml: here')
    with pytest.raises(InvalidConfiguration):
        get_config(test_config_path)

    # Test case 3: Non-dict YAML config
    with open(test_config_path, 'w', encoding='utf-8') as f:
        f.write('- item1\n- item2')
    with pytest.raises(InvalidConfiguration):
        get_config(test_config_path)

    # Test case 4: Non-existent config file
    non_existent_path = os.path.join(temp_dir, 'nonexistent.yml')
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_path)

    # Clean up
    shutil.rmtree(temp_dir)


# LLM-generated content at query #23
#--------------------------

# Unit test for function get_user_config
def test_get_user_config():
    # Test default_config=True
    assert get_user_config(default_config=True) == DEFAULT_CONFIG

    # Test default_config as a dict
    custom_config = {'cookiecutters_dir': '/custom/path'}
    expected_config = merge_configs(DEFAULT_CONFIG, custom_config)
    assert get_user_config(default_config=custom_config) == expected_config

    # Test with a custom config file
    # Mocking file existence and content would be needed for a full test
    # This is a placeholder for that logic

    # Test with COOKIECUTTER_CONFIG environment variable
    # Mocking environment variable would be needed for a full test
    # This is a placeholder for that logic

    # Test default behavior (no config file)
    assert get_user_config() == DEFAULT_CONFIG


# LLM-generated content at query #24
#--------------------------

# Unit test for function get_config
def test_get_config():
    # Test case 1: Config file exists and is valid
    valid_config_path = 'tests/test-configs/valid_config.yml'
    config = get_config(valid_config_path)
    assert isinstance(config, dict)
    assert 'cookiecutters_dir' in config
    assert 'replay_dir' in config
    assert 'default_context' in config
    assert 'abbreviations' in config

    # Test case 2: Config file does not exist
    invalid_config_path = 'tests/test-configs/nonexistent_config.yml'
    try:
        get_config(invalid_config_path)
    except ConfigDoesNotExistException:
        pass
    else:
        assert False, "Expected ConfigDoesNotExistException"

    # Test case 3: Config file is invalid YAML
    invalid_yaml_path = 'tests/test-configs/invalid_yaml_config.yml'
    try:
        get_config(invalid_yaml_path)
    except InvalidConfiguration:
        pass
    else:
        assert False, "Expected InvalidConfiguration"

    # Test case 4: Config file is valid YAML but top-level element is not a dict
    invalid_top_level_path = 'tests/test-configs/invalid_top_level_config.yml'
    try:
        get_config(invalid_top_level_path)
    except InvalidConfiguration:
        pass
    else:
        assert False, "Expected InvalidConfiguration"


# LLM-generated content at query #25
#--------------------------

# Unit test for function get_config
def test_get_config():
    """Test the get_config function."""
    import tempfile
    import pytest
    from pathlib import Path

    # Test with valid YAML
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp_file:
        tmp_file.write('cookiecutters_dir: /tmp/cookiecutters\n')
        tmp_file.write('replay_dir: /tmp/replay\n')
        tmp_file.write('default_context:\n')
        tmp_file.write('  project_name: My Project\n')
        tmp_file.write('abbreviations:\n')
        tmp_file.write('  gh: https://github.com/{0}.git\n')
        tmp_file_name = tmp_file.name

    config = get_config(tmp_file_name)
    assert config['cookiecutters_dir'] == '/tmp/cookiecutters'
    assert config['replay_dir'] == '/tmp/replay'
    assert config['default_context']['project_name'] == 'My Project'
    assert config['abbreviations']['gh'] == 'https://github.com/{0}.git'

    # Test with invalid YAML
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp_file:
        tmp_file.write('invalid yaml')
        tmp_file_name = tmp_file.name

    with pytest.raises(InvalidConfiguration):
        get_config(tmp_file_name)

    # Test with non-existing file
    non_existing_file = Path('/tmp/non_existing_file')
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existing_file)

    # Test with non-dict YAML
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp_file:
        tmp_file.write('- item1\n')
        tmp_file.write('- item2\n')
        tmp_file_name = tmp_file.name

    with pytest.raises(InvalidConfiguration):
        get_config(tmp_file_name)


# LLM-generated content at query #26
#--------------------------

# Unit test for function get_config
def test_get_config():
    # Test with a valid config file
    valid_config_path = "tests/files/config_test_valid.yml"
    expected_output = {
        'cookiecutters_dir': os.path.expanduser('~/.cookiecutters/'),
        'replay_dir': os.path.expanduser('~/.cookiecutter_replay/'),
        'default_context': collections.OrderedDict([]),
        'abbreviations': BUILTIN_ABBREVIATIONS,
    }
    assert get_config(valid_config_path) == expected_output

    # Test with a non-existent config file
    invalid_config_path = "tests/files/config_test_invalid.yml"
    try:
        get_config(invalid_config_path)
    except ConfigDoesNotExistException:
        pass
    else:
        assert False, "Expected ConfigDoesNotExistException"

    # Test with an invalid YAML file
    invalid_yaml_path = "tests/files/config_test_invalid_yaml.yml"
    try:
        get_config(invalid_yaml_path)
    except InvalidConfiguration:
        pass
    else:
        assert False, "Expected InvalidConfiguration"

    # Test with a YAML file that is not a dict
    invalid_yaml_path = "tests/files/config_test_not_dict.yml"
    try:
        get_config(invalid_yaml_path)
    except InvalidConfiguration:
        pass
    else:
        assert False, "Expected InvalidConfiguration"



# LLM-generated content at query #27
#--------------------------

# Unit test for function get_config
def test_get_config():
    import tempfile
    import pytest

    # Create a temporary config file
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
        temp_file.write("cookiecutters_dir: /tmp/cookiecutters\n")
        temp_file.write("replay_dir: /tmp/replay\n")
        temp_file.write("default_context:\n")
        temp_file.write("  key1: value1\n")
        temp_file.write("  key2: value2\n")
        temp_file.write("abbreviations:\n")
        temp_file.write("  gh: https://github.com/{0}.git\n")
        temp_file_path = temp_file.name

    # Test successful config loading
    config = get_config(temp_file_path)
    assert config['cookiecutters_dir'] == '/tmp/cookiecutters'
    assert config['replay_dir'] == '/tmp/replay'
    assert config['default_context']['key1'] == 'value1'
    assert config['default_context']['key2'] == 'value2'
    assert config['abbreviations']['gh'] == 'https://github.com/{0}.git'

    # Test non-existent config file
    with pytest.raises(ConfigDoesNotExistException):
        get_config('/non/existent/path')

    # Test invalid YAML file
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
        temp_file.write("invalid: yaml: file\n")
        invalid_yaml_path = temp_file.name

    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_path)

    # Test YAML file with non-dict top-level element
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
        temp_file.write("not a dict\n")
        non_dict_yaml_path = temp_file.name

    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_path)

    # Clean up temporary files
    os.unlink(temp_file_path)
    os.unlink(invalid_yaml_path)
    os.unlink(non_dict_yaml_path)


# LLM-generated content at query #28
#--------------------------

# Unit test for function get_config
def test_get_config():
    """Test the get_config function."""
    import tempfile
    import shutil
    import pytest

    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    test_config_path = os.path.join(temp_dir, 'test_config.yml')

    # Test with a valid config file
    valid_config = {
        'cookiecutters_dir': '/custom/cookiecutters',
        'replay_dir': '/custom/replay',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://custom.com/{0}'},
    }
    with open(test_config_path, 'w', encoding='utf-8') as f:
        yaml.dump(valid_config, f)

    config = get_config(test_config_path)
    assert config['cookiecutters_dir'] == '/custom/cookiecutters'
    assert config['replay_dir'] == '/custom/replay'
    assert config['default_context'] == {'key': 'value'}
    assert config['abbreviations']['custom'] == 'https://custom.com/{0}'
    assert config['abbreviations']['gh'] == 'https://github.com/{0}.git'  # default preserved

    # Test with an invalid YAML file
    with open(test_config_path, 'w', encoding='utf-8') as f:
        f.write('invalid: yaml: file')

    with pytest.raises(InvalidConfiguration):
        get_config(test_config_path)

    # Test with a non-dict YAML file
    with open(test_config_path, 'w', encoding='utf-8') as f:
        f.write('- item1\n- item2')

    with pytest.raises(InvalidConfiguration):
        get_config(test_config_path)

    # Test with a non-existent file
    with pytest.raises(ConfigDoesNotExistException):
        get_config(os.path.join(temp_dir, 'nonexistent.yml'))

    # Clean up
    shutil.rmtree(temp_dir)


# LLM-generated content at query #29
#--------------------------

# Unit test for function get_config
def test_get_config():
    test_config_path = "test_config.yaml"
    test_dict = {
        "cookiecutters_dir": "~/test_dir",
        "replay_dir": "~/test_replay",
        "default_context": {},
        "abbreviations": {"gh": "https://github.com/{0}.git"}
    }

    with open(test_config_path, "w") as file:
        yaml.dump(test_dict, file)

    config = get_config(test_config_path)

    assert config["cookiecutters_dir"] == os.path.expanduser("~/test_dir")
    assert config["replay_dir"] == os.path.expanduser("~/test_replay")
    assert config["abbreviations"] == {"gh": "https://github.com/{0}.git"}

    os.remove(test_config_path)



# LLM-generated content at query #30
#--------------------------

# Unit test for function get_user_config
def test_get_user_config():
    # Test default config
    assert get_user_config(default_config=True) == DEFAULT_CONFIG

    # Test custom default config
    custom_default = {'cookiecutters_dir': '/custom/path'}
    expected = merge_configs(DEFAULT_CONFIG, custom_default)
    assert get_user_config(default_config=custom_default) == expected

    # Test with non-existent config file (should raise ConfigDoesNotExistException)
    try:
        get_user_config(config_file='/nonexistent/path')
        assert False, "Expected ConfigDoesNotExistException"
    except ConfigDoesNotExistException:
        pass

    # Test with invalid YAML file (should raise InvalidConfiguration)
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml') as tmp:
        tmp.write('invalid: yaml: file')
        tmp.flush()
        try:
            get_user_config(config_file=tmp.name)
            assert False, "Expected InvalidConfiguration"
        except InvalidConfiguration:
            pass

    # Test with valid config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml') as tmp:
        tmp.write('cookiecutters_dir: /custom/path')
        tmp.flush()
        result = get_user_config(config_file=tmp.name)
        assert result['cookiecutters_dir'] == '/custom/path'

    # Test environment variable override
    import os
    os.environ['COOKIECUTTER_CONFIG'] = USER_CONFIG_PATH
    try:
        # Should use the environment variable path
        result = get_user_config()
        assert result == get_config(USER_CONFIG_PATH)
    finally:
        del os.environ['COOKIECUTTER_CONFIG']

    # Test default behavior (no config file, no env var)
    result = get_user_config()
    if os.path.exists(USER_CONFIG_PATH):
        assert result == get_config(USER_CONFIG_PATH)
    else:
        assert result == DEFAULT_CONFIG


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

# Unit test for function get_config
def test_get_config():
    import tempfile

    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
        temp_file.write('cookiecutters_dir: /tmp/cookiecutters\n')
        temp_file.write('replay_dir: /tmp/replay\n')
        temp_file.write('default_context:\n')
        temp_file.write('  key1: value1\n')
        temp_file.write('abbreviations:\n')
        temp_file.write('  gh: https://github.com/{0}.git\n')
        temp_file_path = temp_file.name

    # Test the function
    config = get_config(temp_file_path)
    assert config['cookiecutters_dir'] == '/tmp/cookiecutters'
    assert config['replay_dir'] == '/tmp/replay'
    assert config['default_context'] == {'key1': 'value1'}
    assert config['abbreviations']['gh'] == 'https://github.com/{0}.git'

    # Clean up
    os.unlink(temp_file_path)


# LLM-generated content at query #2
#--------------------------

# Unit test for function get_user_config
def test_get_user_config():
    # Test case 1: default_config is True
    assert get_user_config(default_config=True) == DEFAULT_CONFIG

    # Test case 2: default_config is a dict
    custom_config = {'cookiecutters_dir': '/custom/path'}
    expected = merge_configs(DEFAULT_CONFIG, custom_config)
    assert get_user_config(default_config=custom_config) == expected

    # Test case 3: config_file is specified
    # Mocking file existence and content would be needed here
    # This is a placeholder test case
    try:
        get_user_config(config_file='/nonexistent/path')
        assert False, "Expected ConfigDoesNotExistException"
    except ConfigDoesNotExistException:
        pass

    # Test case 4: COOKIECUTTER_CONFIG environment variable is set
    # Mocking environment variables would be needed here
    # This is a placeholder test case
    try:
        os.environ['COOKIECUTTER_CONFIG'] = '/nonexistent/path'
        get_user_config()
        assert False, "Expected ConfigDoesNotExistException"
    except ConfigDoesNotExistException:
        pass
    finally:
        os.environ.pop('COOKIECUTTER_CONFIG', None)

    # Test case 5: Default user config exists
    # Mocking file existence and content would be needed here
    # This is a placeholder test case
    try:
        get_user_config()
    except Exception:
        assert False, "Unexpected exception"

    print("All tests passed!")

if __name__ == '__main__':
    test_get_user_config()


# LLM-generated content at query #3
#--------------------------

# Unit test for function get_config
def test_get_config():
    """Test the get_config function."""
    import tempfile
    import shutil
    from pathlib import Path

    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    try:
        # Create a test config file
        config_path = Path(temp_dir) / 'test_config.yml'
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write('''
cookiecutters_dir: /custom/cookiecutters/
replay_dir: /custom/replay/
default_context:
    key1: value1
abbreviations:
    custom: https://custom.com/{0}
            ''')

        # Test loading the config
        config = get_config(config_path)
        assert config['cookiecutters_dir'] == '/custom/cookiecutters/'
        assert config['replay_dir'] == '/custom/replay/'
        assert config['default_context']['key1'] == 'value1'
        assert config['abbreviations']['custom'] == 'https://custom.com/{0}'
        assert config['abbreviations']['gh'] == 'https://github.com/{0}.git'  # Built-in should remain

        # Test with invalid YAML
        invalid_config_path = Path(temp_dir) / 'invalid_config.yml'
        with open(invalid_config_path, 'w', encoding='utf-8') as f:
            f.write('invalid: yaml: file')
        
        try:
            get_config(invalid_config_path)
            assert False, "Expected InvalidConfiguration exception"
        except InvalidConfiguration:
            pass

        # Test with non-dict YAML
        non_dict_config_path = Path(temp_dir) / 'non_dict_config.yml'
        with open(non_dict_config_path, 'w', encoding='utf-8') as f:
            f.write('just a string')
        
        try:
            get_config(non_dict_config_path)
            assert False, "Expected InvalidConfiguration exception"
        except InvalidConfiguration:
            pass

    finally:
        # Clean up
        shutil.rmtree(temp_dir)


# LLM-generated content at query #4
#--------------------------

# Unit test for function get_user_config
def test_get_user_config():
    # Case 1: default_config is True
    assert get_user_config(default_config=True) == DEFAULT_CONFIG

    # Case 2: default_config is a dict
    custom_config = {'cookiecutters_dir': '/custom/path'}
    expected_config = DEFAULT_CONFIG.copy()
    expected_config.update(custom_config)
    assert get_user_config(default_config=custom_config) == expected_config

    # Case 3: config_file is provided and different from USER_CONFIG_PATH
    # Mocking file existence and content for this case
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
        temp_file.write(yaml.dump({'cookiecutters_dir': '/custom/path'}))
        temp_file_path = temp_file.name
    
    assert get_user_config(config_file=temp_file_path)['cookiecutters_dir'] == '/custom/path'
    os.unlink(temp_file_path)

    # Case 4: COOKIECUTTER_CONFIG environment variable is set
    # Mocking file existence and content for this case
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
        temp_file.write(yaml.dump({'cookiecutters_dir': '/env/path'}))
        temp_file_path = temp_file.name
    
    os.environ['COOKIECUTTER_CONFIG'] = temp_file_path
    assert get_user_config()['cookiecutters_dir'] == '/env/path'
    os.unlink(temp_file_path)
    del os.environ['COOKIECUTTER_CONFIG']

    # Case 5: Default user config file exists
    # Mocking file existence and content for this case
    with open(USER_CONFIG_PATH, 'w') as user_config_file:
        user_config_file.write(yaml.dump({'cookiecutters_dir': '/user/path'}))
    
    assert get_user_config()['cookiecutters_dir'] == '/user/path'
    os.unlink(USER_CONFIG_PATH)

    # Case 6: No config file exists, return defaults
    assert get_user_config() == DEFAULT_CONFIG


# LLM-generated content at query #5
#--------------------------

# Unit test for function get_config
def test_get_config():
    import tempfile
    import pytest

    # Test case 1: Valid YAML file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml') as tmp:
        tmp.write('cookiecutters_dir: /tmp/cookies\nreplay_dir: /tmp/replay')
        tmp.flush()
        config = get_config(tmp.name)
        assert config['cookiecutters_dir'] == '/tmp/cookies'
        assert config['replay_dir'] == '/tmp/replay'

    # Test case 2: Invalid YAML file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml') as tmp:
        tmp.write('invalid: yaml: file')
        tmp.flush()
        with pytest.raises(InvalidConfiguration):
            get_config(tmp.name)

    # Test case 3: Non-existent file
    with pytest.raises(ConfigDoesNotExistException):
        get_config('/nonexistent/file.yaml')

    # Test case 4: YAML file with environment variables
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml') as tmp:
        tmp.write('cookiecutters_dir: $HOME/cookies\nreplay_dir: ~/replay')
        tmp.flush()
        config = get_config(tmp.name)
        assert config['cookiecutters_dir'] == os.path.expandvars('$HOME/cookies')
        assert config['replay_dir'] == os.path.expanduser('~/replay')

    # Test case 5: YAML file with non-dict top-level element
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml') as tmp:
        tmp.write('- item1\n- item2')
        tmp.flush()
        with pytest.raises(InvalidConfiguration):
            get_config(tmp.name)

    # Test case 6: Empty YAML file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml') as tmp:
        tmp.write('')
        tmp.flush()
        config = get_config(tmp.name)
        assert config == DEFAULT_CONFIG


# LLM-generated content at query #6
#--------------------------

# Unit test for function get_config
def test_get_config():
    """Test the get_config function."""
    import tempfile
    import pytest

    # Test with valid YAML file
    valid_yaml = '''
    cookiecutters_dir: /custom/cookiecutters/
    replay_dir: /custom/replay/
    default_context:
        full_name: "Test User"
    abbreviations:
        gh: "https://github.com/{0}.git"
    '''
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml') as tmp:
        tmp.write(valid_yaml)
        tmp.flush()
        config = get_config(tmp.name)
        assert config['cookiecutters_dir'] == '/custom/cookiecutters/'
        assert config['replay_dir'] == '/custom/replay/'
        assert config['default_context']['full_name'] == 'Test User'
        assert config['abbreviations']['gh'] == 'https://github.com/{0}.git'

    # Test with invalid YAML file
    invalid_yaml = 'invalid: yaml: file'
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml') as tmp:
        tmp.write(invalid_yaml)
        tmp.flush()
        with pytest.raises(InvalidConfiguration):
            get_config(tmp.name)

    # Test with non-existent file
    with pytest.raises(ConfigDoesNotExistException):
        get_config('/nonexistent/path/config.yaml')


# LLM-generated content at query #7
#--------------------------

# Unit test for function get_config
def test_get_config():
    # Create a temporary config file
    import tempfile
    import pathlib
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
        temp_file.write("cookiecutters_dir: /tmp/cookiecutters\n")
        temp_file.write("replay_dir: /tmp/cookiecutter_replay\n")
        temp_file.write("default_context: {'key': 'value'}\n")
        temp_file.write("abbreviations: {'gh': 'https://github.com/{0}.git'}\n")
        temp_file_path = pathlib.Path(temp_file.name)

    # Test that the config file is correctly loaded
    config = get_config(temp_file_path)
    assert config['cookiecutters_dir'] == '/tmp/cookiecutters'
    assert config['replay_dir'] == '/tmp/cookiecutter_replay'
    assert config['default_context'] == {'key': 'value'}
    assert config['abbreviations'] == {'gh': 'https://github.com/{0}.git'}

    # Test that an invalid config file raises an exception
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
        temp_file.write("invalid yaml")
        temp_file_path = pathlib.Path(temp_file.name)
    try:
        get_config(temp_file_path)
        assert False, "Expected InvalidConfiguration exception"
    except InvalidConfiguration:
        pass

    # Clean up temporary files
    os.unlink(temp_file_path)


# LLM-generated content at query #8
#--------------------------

# Unit test for function get_config
def test_get_config():
    import tempfile
    import pytest

    # Test with a valid config file
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as config_file:
        config_file.write('cookiecutters_dir: /custom/path')
        config_file_path = config_file.name

    config = get_config(config_file_path)
    assert config['cookiecutters_dir'] == '/custom/path'

    # Test with an invalid YAML file
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as config_file:
        config_file.write('invalid yaml')
        config_file_path = config_file.name

    with pytest.raises(InvalidConfiguration):
        get_config(config_file_path)

    # Test with a non-existent config file
    with pytest.raises(ConfigDoesNotExistException):
        get_config('/non/existent/path')

    # Clean up
    os.remove(config_file_path)


# LLM-generated content at query #9
#--------------------------

# Unit test for function get_user_config
def test_get_user_config():
    # Test with default_config=True
    assert get_user_config(default_config=True) == DEFAULT_CONFIG

    # Test with default_config as a dict
    custom_config = {'cookiecutters_dir': '/custom/path'}
    expected = merge_configs(DEFAULT_CONFIG, custom_config)
    assert get_user_config(default_config=custom_config) == expected

    # Test with custom config file (mock this by creating a temporary file)
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp:
        tmp.write('cookiecutters_dir: /tmp/custom\n')
        tmp_path = tmp.name

    try:
        assert get_user_config(config_file=tmp_path)['cookiecutters_dir'] == '/tmp/custom'
    finally:
        os.unlink(tmp_path)

    # Test with COOKIECUTTER_CONFIG environment variable
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp:
        tmp.write('cookiecutters_dir: /tmp/env\n')
        tmp_path = tmp.name

    try:
        import os
        os.environ['COOKIECUTTER_CONFIG'] = tmp_path
        assert get_user_config()['cookiecutters_dir'] == os.path.expanduser('/tmp/env')
    finally:
        os.unlink(tmp_path)
        del os.environ['COOKIECUTTER_CONFIG']

    # Test with default user config file (mock this by creating ~/.cookiecutterrc)
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', dir=os.path.expanduser('~'), prefix='.cookiecutterrc') as tmp:
        tmp.write('cookiecutters_dir: /tmp/home\n')
        tmp_path = tmp.name
        try:
            assert get_user_config()['cookiecutters_dir'] == os.path.expanduser('/tmp/home')
        finally:
            os.unlink(tmp_path)

    # Test fallback to default config when no config files exist
    assert get_user_config() == DEFAULT_CONFIG


# LLM-generated content at query #10
#--------------------------

# Unit test for function get_config
def test_get_config():
    import tempfile
    import pytest
    from pathlib import Path

    # Create a temporary YAML file for testing
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp_file:
        yaml_content = """
        cookiecutters_dir: /custom/cookiecutters
        replay_dir: /custom/replay
        default_context:
            project_name: TestProject
        abbreviations:
            gh: https://custom.github.com/{0}.git
        """
        tmp_file.write(yaml_content)
        tmp_file_path = tmp_file.name

    try:
        # Test loading the config from the temporary file
        config = get_config(tmp_file_path)
        assert config['cookiecutters_dir'] == '/custom/cookiecutters'
        assert config['replay_dir'] == '/custom/replay'
        assert config['default_context']['project_name'] == 'TestProject'
        assert config['abbreviations']['gh'] == 'https://custom.github.com/{0}.git'

        # Test handling of non-existent file
        with pytest.raises(ConfigDoesNotExistException):
            get_config('/non/existent/path')

        # Test handling of invalid YAML
        with open(tmp_file_path, 'w') as tmp_file:
            tmp_file.write('invalid: yaml: content')
        with pytest.raises(InvalidConfiguration):
            get_config(tmp_file_path)

        # Test handling of non-dict YAML
        with open(tmp_file_path, 'w') as tmp_file:
            tmp_file.write('not a dict')
        with pytest.raises(InvalidConfiguration):
            get_config(tmp_file_path)

    finally:
        # Clean up the temporary file
        Path(tmp_file_path).unlink()


# LLM-generated content at query #11
#--------------------------

# Unit test for function get_config
def test_get_config():
    import tempfile
    import pytest

    # Test with a valid config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml') as tmp:
        tmp.write('cookiecutters_dir: /tmp/cookies\nreplay_dir: /tmp/replay')
        tmp.flush()
        config = get_config(tmp.name)
        assert config['cookiecutters_dir'] == '/tmp/cookies'
        assert config['replay_dir'] == '/tmp/replay'

    # Test with a non-existent file
    with pytest.raises(ConfigDoesNotExistException):
        get_config('/nonexistent/path')

    # Test with invalid YAML
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml') as tmp:
        tmp.write('invalid: yaml: here')
        tmp.flush()
        with pytest.raises(InvalidConfiguration):
            get_config(tmp.name)

    # Test with non-dict YAML
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml') as tmp:
        tmp.write('- list\n- item')
        tmp.flush()
        with pytest.raises(InvalidConfiguration):
            get_config(tmp.name)


# LLM-generated content at query #12
#--------------------------

# Unit test for function get_config
def test_get_config():
    """Test get_config function."""
    # Create a temporary config file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml') as config_file:
        config_file.write('cookiecutters_dir: /tmp/cookiecutters\n')
        config_file.write('replay_dir: /tmp/replay\n')
        config_file.flush()

        # Test loading config from file
        config = get_config(config_file.name)
        assert config['cookiecutters_dir'] == '/tmp/cookiecutters'
        assert config['replay_dir'] == '/tmp/replay'

        # Test merging with default config
        assert config['default_context'] == collections.OrderedDict([])
        assert config['abbreviations'] == BUILTIN_ABBREVIATIONS

    # Test invalid YAML file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml') as bad_config_file:
        bad_config_file.write('invalid: yaml: file\n')
        bad_config_file.flush()
        try:
            get_config(bad_config_file.name)
            assert False, "Should have raised InvalidConfiguration"
        except InvalidConfiguration:
            pass

    # Test non-existent file
    try:
        get_config('/nonexistent/file')
        assert False, "Should have raised ConfigDoesNotExistException"
    except ConfigDoesNotExistException:
        pass


# LLM-generated content at query #13
#--------------------------

# Unit test for function get_config
def test_get_config():
    """Test the get_config function."""
    import tempfile
    import pytest

    # Test with valid YAML file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml') as tmp:
        tmp.write('cookiecutters_dir: /custom/path\n')
        tmp.write('replay_dir: /another/path\n')
        tmp.flush()
        config = get_config(tmp.name)
        assert config['cookiecutters_dir'] == '/custom/path'
        assert config['replay_dir'] == '/another/path'

    # Test with invalid YAML file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml') as tmp:
        tmp.write('invalid: yaml: file\n')
        tmp.flush()
        with pytest.raises(InvalidConfiguration):
            get_config(tmp.name)

    # Test with non-existent file
    with pytest.raises(ConfigDoesNotExistException):
        get_config('/nonexistent/path')


# LLM-generated content at query #14
#--------------------------

# Unit test for function get_config
def test_get_config():
    import tempfile
    import pytest

    # Test with a valid YAML file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml') as tmp:
        tmp.write('cookiecutters_dir: /custom/path\n')
        tmp.write('replay_dir: /another/path\n')
        tmp.write('default_context:\n  key1: value1\n')
        tmp.write('abbreviations:\n  custom: https://example.com/{0}.git\n')
        tmp.flush()

        config = get_config(tmp.name)
        assert config['cookiecutters_dir'] == '/custom/path'
        assert config['replay_dir'] == '/another/path'
        assert config['default_context']['key1'] == 'value1'
        assert config['abbreviations']['custom'] == 'https://example.com/{0}.git'
        assert 'gh' in config['abbreviations']  # Should preserve builtins

    # Test with non-existent file
    with pytest.raises(ConfigDoesNotExistException):
        get_config('/nonexistent/path')

    # Test with invalid YAML
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml') as tmp:
        tmp.write('invalid: yaml: here')
        tmp.flush()
        with pytest.raises(InvalidConfiguration):
            get_config(tmp.name)

    # Test with non-dict YAML
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml') as tmp:
        tmp.write('- item1\n- item2')
        tmp.flush()
        with pytest.raises(InvalidConfiguration):
            get_config(tmp.name)

    # Test path expansion
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml') as tmp:
        tmp.write('cookiecutters_dir: ~/custom\n')
        tmp.write('replay_dir: $HOME/replay\n')
        tmp.flush()
        config = get_config(tmp.name)
        assert config['cookiecutters_dir'] == os.path.expanduser('~/custom')
        assert config['replay_dir'] == os.path.expandvars('$HOME/replay')


# LLM-generated content at query #15
#--------------------------

# Unit test for function get_config
def test_get_config():
    import tempfile
    import shutil
    import pytest

    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()

    try:
        # Create a temporary config file
        config_file_path = os.path.join(temp_dir, 'test_config.yml')
        with open(config_file_path, 'w', encoding='utf-8') as file:
            file.write('cookiecutters_dir: /tmp/cookiecutters\nreplay_dir: /tmp/replay\n')

        # Test get_config with the temporary config file
        config = get_config(config_file_path)
        assert config['cookiecutters_dir'] == '/tmp/cookiecutters'
        assert config['replay_dir'] == '/tmp/replay'

        # Test get_config with a non-existent file
        with pytest.raises(ConfigDoesNotExistException):
            get_config('/path/to/nonexistent/config.yml')

        # Test get_config with an invalid YAML file
        invalid_yaml_path = os.path.join(temp_dir, 'invalid.yml')
        with open(invalid_yaml_path, 'w', encoding='utf-8') as file:
            file.write('invalid yaml content')
        with pytest.raises(InvalidConfiguration):
            get_config(invalid_yaml_path)

        # Test get_config with a YAML file that has a top-level element that is not a dict
        invalid_top_level_path = os.path.join(temp_dir, 'invalid_top_level.yml')
        with open(invalid_top_level_path, 'w', encoding='utf-8') as file:
            file.write('- not a dict')
        with pytest.raises(InvalidConfiguration):
            get_config(invalid_top_level_path)

    finally:
        # Clean up the temporary directory
        shutil.rmtree(temp_dir)


# LLM-generated content at query #16
#--------------------------

# Unit test for function get_user_config
def test_get_user_config():
    # Test with default_config=True
    assert get_user_config(default_config=True) == DEFAULT_CONFIG

    # Test with default_config as a dict
    custom_config = {'cookiecutters_dir': '/custom/path'}
    expected_config = DEFAULT_CONFIG.copy()
    expected_config.update(custom_config)
    assert get_user_config(default_config=custom_config) == expected_config

    # Test with custom config file
    # Assuming 'test_config.yml' exists and contains valid YAML config
    # Note: This requires a mock or a real file for testing
    # assert get_user_config(config_file='test_config.yml') == expected_config_from_file

    # Test with environment variable COOKIECUTTER_CONFIG
    # Note: This requires mocking os.environ and the existence of the config file
    # os.environ['COOKIECUTTER_CONFIG'] = 'test_config.yml'
    # assert get_user_config() == expected_config_from_file
    # del os.environ['COOKIECUTTER_CONFIG']

    # Test with default USER_CONFIG_PATH
    # Note: This requires the existence of the default config file
    # assert get_user_config() == expected_config_from_default_file

    # Test when no config file exists
    # Note: This requires ensuring USER_CONFIG_PATH does not exist
    # assert get_user_config() == DEFAULT_CONFIG

    print("All tests passed.")

# Run the unit test
test_get_user_config()


# LLM-generated content at query #17
#--------------------------

# Unit test for function get_config
def test_get_config():
    """Test the get_config function."""
    # Test with a valid config file
    valid_config = {
        'cookiecutters_dir': '/custom/cookiecutters',
        'replay_dir': '/custom/replay',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://custom.com/{0}.git'},
    }
    with open('valid_config.yml', 'w', encoding='utf-8') as f:
        yaml.dump(valid_config, f)
    
    config = get_config('valid_config.yml')
    assert config['cookiecutters_dir'] == '/custom/cookiecutters'
    assert config['replay_dir'] == '/custom/replay'
    assert config['default_context'] == {'key': 'value'}
    assert config['abbreviations'] == {
        'gh': 'https://github.com/{0}.git',
        'gl': 'https://gitlab.com/{0}.git',
        'bb': 'https://bitbucket.org/{0}',
        'custom': 'https://custom.com/{0}.git',
    }
    
    # Test with an invalid YAML file
    with open('invalid_config.yml', 'w', encoding='utf-8') as f:
        f.write('invalid: yaml: file')
    
    try:
        get_config('invalid_config.yml')
        assert False, "Expected InvalidConfiguration exception"
    except InvalidConfiguration:
        pass
    
    # Test with a non-existent file
    try:
        get_config('nonexistent.yml')
        assert False, "Expected ConfigDoesNotExistException"
    except ConfigDoesNotExistException:
        pass
    
    # Clean up
    os.remove('valid_config.yml')
    os.remove('invalid_config.yml')


# LLM-generated content at query #18
#--------------------------

# Unit test for function get_user_config
def test_get_user_config():
    # Test default config
    assert get_user_config(default_config=True) == DEFAULT_CONFIG

    # Test custom default config
    custom_default = {'cookiecutters_dir': '/custom/path'}
    expected = merge_configs(DEFAULT_CONFIG, custom_default)
    assert get_user_config(default_config=custom_default) == expected

    # Test with non-existent config file (should raise ConfigDoesNotExistException)
    try:
        get_user_config(config_file='/nonexistent/path')
        assert False, "Expected ConfigDoesNotExistException"
    except ConfigDoesNotExistException:
        pass

    # Test with invalid YAML config file (should raise InvalidConfiguration)
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml') as tmp:
        tmp.write('invalid yaml: [')
        tmp.flush()
        try:
            get_user_config(config_file=tmp.name)
            assert False, "Expected InvalidConfiguration"
        except InvalidConfiguration:
            pass

    # Test with valid config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml') as tmp:
        tmp.write('cookiecutters_dir: /custom/path\nreplay_dir: /custom/replay')
        tmp.flush()
        result = get_user_config(config_file=tmp.name)
        assert result['cookiecutters_dir'] == '/custom/path'
        assert result['replay_dir'] == '/custom/replay'

    # Test environment variable config
    import os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml') as tmp:
        tmp.write('cookiecutters_dir: /env/path')
        tmp.flush()
        os.environ['COOKIECUTTER_CONFIG'] = tmp.name
        result = get_user_config()
        assert result['cookiecutters_dir'] == '/env/path'
        del os.environ['COOKIECUTTER_CONFIG']

    # Test default user config (when no config exists)
    # Mock os.path.exists to return False for USER_CONFIG_PATH
    original_exists = os.path.exists
    os.path.exists = lambda path: False
    assert get_user_config() == DEFAULT_CONFIG
    os.path.exists = original_exists


# LLM-generated content at query #19
#--------------------------

# Unit test for function get_config
def test_get_config():
    # Test 1: Test with a valid config file
    config_path = "test_config.yaml"
    config_dict = {
        'cookiecutters_dir': '~/custom_cookiecutters/',
        'replay_dir': '~/custom_replay/',
        'default_context': {'project_name': 'Test Project'},
        'abbreviations': {'custom': 'https://custom.com/{0}.git'}
    }
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f)
    
    try:
        result = get_config(config_path)
        assert result['cookiecutters_dir'] == os.path.expanduser('~/custom_cookiecutters/')
        assert result['replay_dir'] == os.path.expanduser('~/custom_replay/')
        assert result['default_context'] == {'project_name': 'Test Project'}
        assert result['abbreviations']['custom'] == 'https://custom.com/{0}.git'
        assert result['abbreviations']['gh'] == 'https://github.com/{0}.git'
    finally:
        os.remove(config_path)

    # Test 2: Test with a non-existent config file
    config_path = "non_existent_config.yaml"
    try:
        get_config(config_path)
        assert False, "Expected ConfigDoesNotExistException"
    except ConfigDoesNotExistException:
        pass

    # Test 3: Test with an invalid YAML file
    config_path = "invalid_config.yaml"
    with open(config_path, 'w') as f:
        f.write("invalid yaml content")
    
    try:
        get_config(config_path)
        assert False, "Expected InvalidConfiguration"
    except InvalidConfiguration:
        pass
    finally:
        os.remove(config_path)

    # Test 4: Test with a non-dict YAML file
    config_path = "non_dict_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump("not a dict", f)
    
    try:
        get_config(config_path)
        assert False, "Expected InvalidConfiguration"
    except InvalidConfiguration:
        pass
    finally:
        os.remove(config_path)


# LLM-generated content at query #20
#--------------------------

# Unit test for function get_config
def test_get_config():
    import tempfile
    import pytest
    from pathlib import Path

    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_config:
        config_content = """
        cookiecutters_dir: /custom/cookiecutters/
        replay_dir: /custom/replay/
        default_context:
            key1: value1
        abbreviations:
            custom: https://custom.com/{0}.git
        """
        temp_config.write(config_content)
        temp_config_path = Path(temp_config.name)

    # Test loading the config
    config = get_config(temp_config_path)

    # Check if the values are correctly loaded and expanded
    assert config['cookiecutters_dir'] == '/custom/cookiecutters/'
    assert config['replay_dir'] == '/custom/replay/'
    assert config['default_context'] == collections.OrderedDict([('key1', 'value1')])
    assert config['abbreviations']['custom'] == 'https://custom.com/{0}.git'

    # Test with non-existent file
    with pytest.raises(ConfigDoesNotExistException):
        get_config(Path('/nonexistent/path'))

    # Test with invalid YAML content
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_invalid_config:
        temp_invalid_config.write("invalid: yaml: content")
        temp_invalid_config_path = Path(temp_invalid_config.name)

    with pytest.raises(InvalidConfiguration):
        get_config(temp_invalid_config_path)

    # Test with invalid top-level element (not a dict)
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_invalid_top_config:
        temp_invalid_top_config.write("- item1\n- item2")
        temp_invalid_top_config_path = Path(temp_invalid_top_config.name)

    with pytest.raises(InvalidConfiguration):
        get_config(temp_invalid_top_config_path)

    # Clean up temporary files
    temp_config_path.unlink()
    temp_invalid_config_path.unlink()
    temp_invalid_top_config_path.unlink()


# LLM-generated content at query #21
#--------------------------

# Unit test for function get_config
def test_get_config():
    # Test valid YAML config
    config_path = "test_config.yaml"
    with open(config_path, "w") as f:
        f.write("cookiecutters_dir: /custom/path\nreplay_dir: /another/path")
    config = get_config(config_path)
    assert config["cookiecutters_dir"] == "/custom/path"
    assert config["replay_dir"] == "/another/path"
    os.remove(config_path)

    # Test invalid YAML config
    config_path = "invalid_config.yaml"
    with open(config_path, "w") as f:
        f.write("invalid: yaml: file")
    try:
        get_config(config_path)
    except InvalidConfiguration:
        pass
    else:
        assert False, "Expected InvalidConfiguration exception"
    os.remove(config_path)

    # Test non-existent config file
    try:
        get_config("nonexistent.yaml")
    except ConfigDoesNotExistException:
        pass
    else:
        assert False, "Expected ConfigDoesNotExistException"



# LLM-generated content at query #22
#--------------------------

# Unit test for function get_user_config
def test_get_user_config():
    # Test case 1: default_config is True
    assert get_user_config(default_config=True) == DEFAULT_CONFIG

    # Test case 2: default_config is a dict
    custom_config = {'cookiecutters_dir': '/custom/path'}
    expected = merge_configs(DEFAULT_CONFIG, custom_config)
    assert get_user_config(default_config=custom_config) == expected

    # Test case 3: config_file is provided
    # Mocking a config file would be needed here
    # This is just a placeholder test
    try:
        get_user_config(config_file='/nonexistent/path')
        assert False, "Expected ConfigDoesNotExistException"
    except ConfigDoesNotExistException:
        pass

    # Test case 4: COOKIECUTTER_CONFIG environment variable is set
    # Mocking environment variables would be needed here
    # This is just a placeholder test
    try:
        os.environ['COOKIECUTTER_CONFIG'] = '/nonexistent/path'
        get_user_config()
        assert False, "Expected ConfigDoesNotExistException"
    except ConfigDoesNotExistException:
        pass
    finally:
        del os.environ['COOKIECUTTER_CONFIG']

    # Test case 5: Default user config exists
    # Mocking file existence would be needed here
    # This is just a placeholder test
    try:
        open(USER_CONFIG_PATH, 'w').close()
        assert isinstance(get_user_config(), dict)
    finally:
        os.remove(USER_CONFIG_PATH)

    # Test case 6: No config available
    assert get_user_config() == DEFAULT_CONFIG

    print("All tests passed!")

if __name__ == '__main__':
    test_get_user_config()


# LLM-generated content at query #23
#--------------------------

# Unit test for function get_config
def test_get_config():
    # Test with a valid config file
    config_file = 'test_config.yml'
    with open(config_file, 'w') as f:
        f.write('cookiecutters_dir: /tmp/cookiecutters\n')
        f.write('replay_dir: /tmp/cookiecutter_replay\n')

    config = get_config(config_file)
    assert config['cookiecutters_dir'] == '/tmp/cookiecutters'
    assert config['replay_dir'] == '/tmp/cookiecutter_replay'

    # Test with an invalid config file
    config_file = 'invalid_config.yml'
    with open(config_file, 'w') as f:
        f.write('invalid')

    try:
        config = get_config(config_file)
    except InvalidConfiguration:
        pass
    else:
        assert False, "Expected InvalidConfiguration exception"

    # Clean up test files
    os.remove('test_config.yml')
    os.remove('invalid_config.yml')


# LLM-generated content at query #24
#--------------------------

# Unit test for function get_config
def test_get_config():
    """Test the get_config function."""
    import tempfile
    import shutil
    from pathlib import Path

    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    config_path = Path(temp_dir) / 'test_config.yml'

    try:
        # Test with a valid YAML file
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write('cookiecutters_dir: /custom/cookiecutters\n')
            f.write('replay_dir: /custom/replay\n')
            f.write('default_context:\n')
            f.write('  key1: value1\n')
            f.write('abbreviations:\n')
            f.write('  gh: https://github.com/{0}.git\n')

        config = get_config(config_path)
        assert config['cookiecutters_dir'] == '/custom/cookiecutters'
        assert config['replay_dir'] == '/custom/replay'
        assert config['default_context']['key1'] == 'value1'
        assert config['abbreviations']['gh'] == 'https://github.com/{0}.git'

        # Test with an invalid YAML file
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write('invalid: yaml: file')

        try:
            get_config(config_path)
            assert False, "Expected InvalidConfiguration"
        except InvalidConfiguration:
            pass

        # Test with a non-existent file
        non_existent_path = Path(temp_dir) / 'nonexistent.yml'
        try:
            get_config(non_existent_path)
            assert False, "Expected ConfigDoesNotExistException"
        except ConfigDoesNotExistException:
            pass

    finally:
        # Clean up
        shutil.rmtree(temp_dir)


# LLM-generated content at query #25
#--------------------------

# Unit test for function get_config
def test_get_config():
    import tempfile
    import pytest

    # Test case 1: Valid YAML file
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp_file:
        tmp_file.write("cookiecutters_dir: ~/custom_cookiecutters\nreplay_dir: ~/custom_replay")
        tmp_file_path = tmp_file.name

    config = get_config(tmp_file_path)
    assert config['cookiecutters_dir'] == os.path.expanduser('~/custom_cookiecutters')
    assert config['replay_dir'] == os.path.expanduser('~/custom_replay')

    # Test case 2: Invalid YAML file
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp_file:
        tmp_file.write("invalid: yaml: file")
        tmp_file_path = tmp_file.name

    with pytest.raises(InvalidConfiguration):
        get_config(tmp_file_path)

    # Test case 3: Non-existent file
    non_existent_file = '/path/to/nonexistent/file'
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test case 4: YAML file with top-level element not being a dictionary
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp_file:
        tmp_file.write("- item1\n- item2")
        tmp_file_path = tmp_file.name

    with pytest.raises(InvalidConfiguration):
        get_config(tmp_file_path)

    # Clean up
    os.unlink(tmp_file_path)



# LLM-generated content at query #26
#--------------------------

# Unit test for function get_config
def test_get_config():
    # Create a temporary config file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("cookiecutters_dir: /tmp/cookiecutters\n")
        f.write("replay_dir: /tmp/replay\n")
        config_file = f.name

    # Test loading the config file
    config = get_config(config_file)
    assert config['cookiecutters_dir'] == '/tmp/cookiecutters'
    assert config['replay_dir'] == '/tmp/replay'

    # Test with invalid YAML
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("invalid yaml")
        config_file = f.name
    try:
        get_config(config_file)
        assert False, "Expected InvalidConfiguration exception"
    except InvalidConfiguration:
        pass

    # Test with non-dict YAML
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("invalid")
        config_file = f.name
    try:
        get_config(config_file)
        assert False, "Expected InvalidConfiguration exception"
    except InvalidConfiguration:
        pass

    # Test with non-existent file
    try:
        get_config('/tmp/nonexistent')
        assert False, "Expected ConfigDoesNotExistException exception"
    except ConfigDoesNotExistException:
        pass

    # Clean up
    os.unlink(config_file)


# LLM-generated content at query #27
#--------------------------

# Unit test for function get_user_config
def test_get_user_config():
    import tempfile
    import pytest

    # Test default config
    assert get_user_config(default_config=True) == DEFAULT_CONFIG

    # Test custom config file
    custom_config = {'cookiecutters_dir': '/custom/path'}
    with tempfile.NamedTemporaryFile('w', delete=False) as temp_file:
        yaml.dump(custom_config, temp_file)
        temp_file_path = temp_file.name
    try:
        assert get_user_config(config_file=temp_file_path) == merge_configs(DEFAULT_CONFIG, custom_config)
    finally:
        os.unlink(temp_file_path)

    # Test environment variable config
    env_config = {'replay_dir': '/env/path'}
    with tempfile.NamedTemporaryFile('w', delete=False) as temp_file:
        yaml.dump(env_config, temp_file)
        temp_file_path = temp_file.name
    os.environ['COOKIECUTTER_CONFIG'] = temp_file_path
    try:
        assert get_user_config() == merge_configs(DEFAULT_CONFIG, env_config)
    finally:
        os.unlink(temp_file_path)
        del os.environ['COOKIECUTTER_CONFIG']

    # Test invalid config file
    with tempfile.NamedTemporaryFile('w', delete=False) as temp_file:
        temp_file.write('invalid: yaml: file')
        temp_file_path = temp_file.name
    with pytest.raises(InvalidConfiguration):
        get_user_config(config_file=temp_file_path)
    os.unlink(temp_file_path)

    # Test non-existent config file
    non_existent_path = '/non/existent/path'
    with pytest.raises(ConfigDoesNotExistException):
        get_user_config(config_file=non_existent_path)


# LLM-generated content at query #28
#--------------------------

# Unit test for function get_config
def test_get_config():
    """Test the get_config function."""
    import tempfile
    import pytest

    # Test valid config
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp_file:
        tmp_file.write("cookiecutters_dir: /tmp/cookiecutters\n")
        tmp_file.write("replay_dir: /tmp/replay\n")
        tmp_file_name = tmp_file.name

    config = get_config(tmp_file_name)
    assert config['cookiecutters_dir'] == '/tmp/cookiecutters'
    assert config['replay_dir'] == '/tmp/replay'

    # Test invalid YAML
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp_file:
        tmp_file.write("invalid: YAML: : syntax\n")
        tmp_file_name = tmp_file.name

    with pytest.raises(InvalidConfiguration):
        get_config(tmp_file_name)

    # Test non-existent file
    with pytest.raises(ConfigDoesNotExistException):
        get_config('/non/existent/path')

    # Test non-dict top-level element
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp_file:
        tmp_file.write("- list\n")
        tmp_file_name = tmp_file.name

    with pytest.raises(InvalidConfiguration):
        get_config(tmp_file_name)

    # Clean up temporary files
    os.unlink(tmp_file_name)


# LLM-generated content at query #29
#--------------------------

# Unit test for function get_config
def test_get_config():
    import tempfile
    import pytest

    # Create a temporary config file
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp_file:
        tmp_file.write('cookiecutters_dir: /tmp/cookiecutters\n')
        tmp_file.write('replay_dir: /tmp/replay\n')
        tmp_file.write('default_context:\n')
        tmp_file.write('  key: value\n')
        tmp_file.write('abbreviations:\n')
        tmp_file.write('  gh: https://github.com/{0}.git\n')
        config_file_path = tmp_file.name

    # Test valid config
    config = get_config(config_file_path)
    assert config['cookiecutters_dir'] == '/tmp/cookiecutters'
    assert config['replay_dir'] == '/tmp/replay'
    assert config['default_context']['key'] == 'value'
    assert config['abbreviations']['gh'] == 'https://github.com/{0}.git'

    # Test non-existent config file
    with pytest.raises(ConfigDoesNotExistException):
        get_config('/non/existent/path')

    # Test invalid YAML
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp_file:
        tmp_file.write('invalid: yaml: here\n')
        invalid_yaml_path = tmp_file.name
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_path)

    # Test non-dict top-level element
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp_file:
        tmp_file.write('- item1\n')
        tmp_file.write('- item2\n')
        non_dict_path = tmp_file.name
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_path)

    # Clean up temporary files
    os.unlink(config_file_path)
    os.unlink(invalid_yaml_path)
    os.unlink(non_dict_path)


# LLM-generated content at query #30
#--------------------------

# Unit test for function get_config
def test_get_config():
    """Test the get_config function."""
    import tempfile
    import shutil
    from pathlib import Path

    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    try:
        # Create a test config file
        config_path = Path(temp_dir) / 'test_config.yml'
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write('cookiecutters_dir: /custom/cookiecutters\n')
            f.write('replay_dir: /custom/replay\n')
            f.write('default_context:\n')
            f.write('  key1: value1\n')
            f.write('abbreviations:\n')
            f.write('  custom: https://custom.com/{0}.git\n')

        # Test loading the config
        config = get_config(config_path)
        assert config['cookiecutters_dir'] == '/custom/cookiecutters'
        assert config['replay_dir'] == '/custom/replay'
        assert config['default_context']['key1'] == 'value1'
        assert config['abbreviations']['custom'] == 'https://custom.com/{0}.git'
        assert config['abbreviations']['gh'] == 'https://github.com/{0}.git'  # Built-in should be preserved

        # Test with non-existent file
        try:
            get_config(Path(temp_dir) / 'nonexistent.yml')
            assert False, "Expected ConfigDoesNotExistException"
        except ConfigDoesNotExistException:
            pass

        # Test with invalid YAML
        invalid_path = Path(temp_dir) / 'invalid.yml'
        with open(invalid_path, 'w', encoding='utf-8') as f:
            f.write('invalid: yaml: file')
        try:
            get_config(invalid_path)
            assert False, "Expected InvalidConfiguration"
        except InvalidConfiguration:
            pass

        # Test with non-dict YAML
        non_dict_path = Path(temp_dir) / 'non_dict.yml'
        with open(non_dict_path, 'w', encoding='utf-8') as f:
            f.write('- item1\n- item2')
        try:
            get_config(non_dict_path)
            assert False, "Expected InvalidConfiguration"
        except InvalidConfiguration:
            pass

    finally:
        # Clean up
        shutil.rmtree(temp_dir)


