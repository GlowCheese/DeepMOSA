####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(
        """
        cookiecutters_dir: /custom/cookiecutters/
        replay_dir: /custom/replay/
        default_context:
            key1: value1
            key2: value2
        abbreviations:
            custom: https://custom.com/{0}
        """
    )

    config = get_config(config_file)
    assert config["cookiecutters_dir"] == "/custom/cookiecutters/"
    assert config["replay_dir"] == "/custom/replay/"
    assert config["default_context"] == {"key1": "value1", "key2": "value2"}
    assert config["abbreviations"]["custom"] == "https://custom.com/{0}"
    assert config["abbreviations"]["gh"] == "https://github.com/{0}.git"

    # Test with environment variables and user home expansion
    os.environ["TEST_DIR"] = "/test/env/dir"
    config_file.write_text(
        """
        cookiecutters_dir: $TEST_DIR/cookiecutters/
        replay_dir: ~/replay/
        """
    )

    config = get_config(config_file)
    assert config["cookiecutters_dir"] == "/test/env/dir/cookiecutters/"
    assert config["replay_dir"] == os.path.expanduser("~/replay/")

    # Test with non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with invalid YAML
    config_file.write_text("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(config_file)

    # Test with non-dict YAML content
    config_file.write_text("not a dict")
    with pytest.raises(InvalidConfiguration):
        get_config(config_file)


# LLM-generated content at query #2
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "test_config.yaml"
    config_content = {
        'cookiecutters_dir': '~/test_cookiecutters/',
        'replay_dir': '~/test_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://custom.com/{0}'}
    }
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config_content, f)

    config = get_config(config_file)
    assert config['cookiecutters_dir'] == os.path.expanduser('~/test_cookiecutters/')
    assert config['replay_dir'] == os.path.expanduser('~/test_replay/')
    assert config['default_context'] == {'key': 'value'}
    assert config['abbreviations'] == {**BUILTIN_ABBREVIATIONS, 'custom': 'https://custom.com/{0}'}

    # Test with a non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with an invalid YAML file
    invalid_yaml_file = tmp_path / "invalid.yaml"
    with open(invalid_yaml_file, 'w', encoding='utf-8') as f:
        f.write("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with a YAML file that is not a dict
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    with open(non_dict_yaml_file, 'w', encoding='utf-8') as f:
        f.write("- list_item1\n- list_item2")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #3
#--------------------------

```python
def test_get_config():
    # Test with a valid config file
    valid_config = {
        'cookiecutters_dir': '~/.test_cookiecutters/',
        'replay_dir': '~/.test_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://custom.com/{0}'}
    }
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(valid_config, f)
        f.flush()
        config = get_config(f.name)
        assert config['cookiecutters_dir'] == os.path.expanduser('~/.test_cookiecutters/')
        assert config['replay_dir'] == os.path.expanduser('~/.test_replay/')
        assert config['default_context'] == {'key': 'value'}
        assert config['abbreviations'] == {**BUILTIN_ABBREVIATIONS, 'custom': 'https://custom.com/{0}'}
        os.unlink(f.name)

    # Test with a non-existent config file
    with pytest.raises(ConfigDoesNotExistException):
        get_config('/non/existent/path.yaml')

    # Test with an invalid YAML file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("invalid: yaml: content: [")
        f.flush()
        with pytest.raises(InvalidConfiguration):
            get_config(f.name)
        os.unlink(f.name)

    # Test with a YAML file that is not a dict
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("- not a dict")
        f.flush()
        with pytest.raises(InvalidConfiguration):
            get_config(f.name)
        os.unlink(f.name)


# LLM-generated content at query #4
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "test_config.yaml"
    config_content = {
        'cookiecutters_dir': '~/test_cookiecutters/',
        'replay_dir': '~/test_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://custom.com/{0}'}
    }
    with open(config_file, 'w') as f:
        yaml.dump(config_content, f)

    result = get_config(config_file)

    assert result['cookiecutters_dir'] == os.path.expanduser('~/test_cookiecutters/')
    assert result['replay_dir'] == os.path.expanduser('~/test_replay/')
    assert result['default_context'] == {'key': 'value'}
    assert result['abbreviations'] == {**BUILTIN_ABBREVIATIONS, 'custom': 'https://custom.com/{0}'}

    # Test with a non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with an invalid YAML file
    invalid_yaml_file = tmp_path / "invalid.yaml"
    with open(invalid_yaml_file, 'w') as f:
        f.write("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with a YAML file that is not a dict
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    with open(non_dict_yaml_file, 'w') as f:
        f.write("- list item 1\n- list item 2")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #5
#--------------------------

```python
def test_get_config():
    # Test with a valid config file
    valid_config = {
        'cookiecutters_dir': '~/.test_cookiecutters/',
        'replay_dir': '~/.test_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'test': 'test_url'}
    }
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(valid_config, f)
        f.flush()
        config = get_config(f.name)
        assert config['cookiecutters_dir'] == os.path.expanduser('~/.test_cookiecutters/')
        assert config['replay_dir'] == os.path.expanduser('~/.test_replay/')
        assert config['default_context'] == {'key': 'value'}
        assert config['abbreviations'] == {**BUILTIN_ABBREVIATIONS, 'test': 'test_url'}
        os.unlink(f.name)

    # Test with a non-existent config file
    with pytest.raises(ConfigDoesNotExistException):
        get_config('/non/existent/path.yaml')

    # Test with an invalid YAML file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("invalid: yaml: content: [")
        f.flush()
        with pytest.raises(InvalidConfiguration):
            get_config(f.name)
        os.unlink(f.name)

    # Test with a YAML file that is not a dict
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("- not a dict")
        f.flush()
        with pytest.raises(InvalidConfiguration):
            get_config(f.name)
        os.unlink(f.name)


# LLM-generated content at query #6
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "config.yaml"
    config_content = {
        'cookiecutters_dir': '~/test_cookiecutters/',
        'replay_dir': '~/test_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://custom.com/{0}'}
    }
    with open(config_file, 'w') as f:
        yaml.dump(config_content, f)

    result = get_config(config_file)

    assert result['cookiecutters_dir'] == os.path.expanduser('~/test_cookiecutters/')
    assert result['replay_dir'] == os.path.expanduser('~/test_replay/')
    assert result['default_context'] == {'key': 'value'}
    assert result['abbreviations'] == {**BUILTIN_ABBREVIATIONS, 'custom': 'https://custom.com/{0}'}

    # Test with a non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with an invalid YAML file
    invalid_yaml_file = tmp_path / "invalid.yaml"
    with open(invalid_yaml_file, 'w') as f:
        f.write("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with a YAML file that is not a dict
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    with open(non_dict_yaml_file, 'w') as f:
        f.write("- list_item1\n- list_item2")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #7
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "config.yaml"
    config_file.write_text("""
cookiecutters_dir: /custom/cookiecutters/
replay_dir: /custom/replay/
default_context:
    key: value
abbreviations:
    custom: https://custom.com/{0}
""")

    config = get_config(config_file)
    assert config['cookiecutters_dir'] == '/custom/cookiecutters/'
    assert config['replay_dir'] == '/custom/replay/'
    assert config['default_context'] == {'key': 'value'}
    assert config['abbreviations']['custom'] == 'https://custom.com/{0}'
    assert config['abbreviations']['gh'] == 'https://github.com/{0}.git'

    # Test with environment variables in paths
    config_file.write_text("""
cookiecutters_dir: $HOME/test_cookiecutters/
replay_dir: ~/test_replay/
""")

    config = get_config(config_file)
    assert config['cookiecutters_dir'] == os.path.expandvars('$HOME/test_cookiecutters/')
    assert config['replay_dir'] == os.path.expanduser('~/test_replay/')

    # Test with non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with invalid YAML
    invalid_yaml_file = tmp_path / "invalid.yaml"
    invalid_yaml_file.write_text("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with non-dict YAML content
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    non_dict_yaml_file.write_text("this is not a dict")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #8
#--------------------------

```python
def test_get_user_config():
    # Test default config
    config = get_user_config(default_config=True)
    assert config == DEFAULT_CONFIG

    # Test custom config dict
    custom_config = {'cookiecutters_dir': '/custom/dir'}
    config = get_user_config(default_config=custom_config)
    expected = merge_configs(DEFAULT_CONFIG, custom_config)
    assert config == expected

    # Test loading from custom config file
    with patch('builtins.open', mock_open(read_data='cookiecutters_dir: /test/dir')):
        with patch('os.path.exists', return_value=True):
            config = get_user_config(config_file='/test/config')
            assert config['cookiecutters_dir'] == '/test/dir'

    # Test COOKIECUTTER_CONFIG environment variable
    with patch.dict('os.environ', {'COOKIECUTTER_CONFIG': '/env/config'}):
        with patch('builtins.open', mock_open(read_data='replay_dir: /env/replay')):
            with patch('os.path.exists', return_value=True):
                config = get_user_config()
                assert config['replay_dir'] == '/env/replay'

    # Test non-existent config file raises exception
    with patch('os.path.exists', return_value=False):
        with pytest.raises(ConfigDoesNotExistException):
            get_user_config(config_file='/nonexistent/config')

    # Test invalid YAML raises exception
    with patch('builtins.open', mock_open(read_data='invalid: yaml: content')):
        with patch('os.path.exists', return_value=True):
            with pytest.raises(InvalidConfiguration):
                get_user_config(config_file='/invalid/config')


# LLM-generated content at query #9
#--------------------------

```python
def test_get_user_config():
    # Test default config
    config = get_user_config(default_config=True)
    assert config == DEFAULT_CONFIG

    # Test custom default config
    custom_default = {'cookiecutters_dir': '/custom/dir'}
    config = get_user_config(default_config=custom_default)
    expected = merge_configs(DEFAULT_CONFIG, custom_default)
    assert config == expected

    # Test loading from custom config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("cookiecutters_dir: /test/dir\nabbreviations:\n  custom: 'test'")
        f.flush()
        config = get_user_config(config_file=f.name)
        assert config['cookiecutters_dir'] == '/test/dir'
        assert config['abbreviations']['custom'] == 'test'
        os.unlink(f.name)

    # Test environment variable config
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("replay_dir: /env/dir")
        f.flush()
        os.environ['COOKIECUTTER_CONFIG'] = f.name
        config = get_user_config()
        assert config['replay_dir'] == '/env/dir'
        del os.environ['COOKIECUTTER_CONFIG']
        os.unlink(f.name)

    # Test user config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("default_context:\n  key: 'value'")
        f.flush()
        os.rename(f.name, USER_CONFIG_PATH)
        config = get_user_config()
        assert config['default_context']['key'] == 'value'
        os.unlink(USER_CONFIG_PATH)


# LLM-generated content at query #10
#--------------------------

```python
def test_get_user_config():
    # Test default config
    config = get_user_config(default_config=True)
    assert config == DEFAULT_CONFIG

    # Test custom config dict
    custom_config = {'cookiecutters_dir': '/custom/dir'}
    config = get_user_config(default_config=custom_config)
    expected = merge_configs(DEFAULT_CONFIG, custom_config)
    assert config == expected

    # Test custom config file
    with patch('builtins.open', mock_open(read_data='cookiecutters_dir: /test/dir')):
        with patch('os.path.exists', return_value=True):
            config = get_user_config(config_file='/test/config')
            assert config['cookiecutters_dir'] == '/test/dir'

    # Test environment variable config
    with patch.dict('os.environ', {'COOKIECUTTER_CONFIG': '/env/config'}):
        with patch('builtins.open', mock_open(read_data='cookiecutters_dir: /env/dir')):
            with patch('os.path.exists', return_value=True):
                config = get_user_config()
                assert config['cookiecutters_dir'] == '/env/dir'

    # Test default user config file
    with patch('builtins.open', mock_open(read_data='cookiecutters_dir: /user/dir')):
        with patch('os.path.exists', return_value=True):
            config = get_user_config()
            assert config['cookiecutters_dir'] == '/user/dir'

    # Test non-existent config file
    with patch('os.path.exists', return_value=False):
        config = get_user_config()
        assert config == DEFAULT_CONFIG

    # Test invalid YAML
    with patch('builtins.open', mock_open(read_data='invalid: yaml: content')):
        with patch('os.path.exists', return_value=True):
            with pytest.raises(InvalidConfiguration):
                get_user_config(config_file='/invalid/config')

    # Test non-dict YAML
    with patch('builtins.open', mock_open(read_data='- list item')):
        with patch('os.path.exists', return_value=True):
            with pytest.raises(InvalidConfiguration):
                get_user_config(config_file='/list/config')


# LLM-generated content at query #11
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "test_config.yaml"
    config_content = {
        'cookiecutters_dir': '~/test_cookiecutters/',
        'replay_dir': '~/test_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://custom.com/{0}'}
    }
    with open(config_file, 'w') as f:
        yaml.dump(config_content, f)

    result = get_config(config_file)
    assert result['cookiecutters_dir'] == os.path.expanduser('~/test_cookiecutters/')
    assert result['replay_dir'] == os.path.expanduser('~/test_replay/')
    assert result['default_context'] == {'key': 'value'}
    assert result['abbreviations'] == {**BUILTIN_ABBREVIATIONS, 'custom': 'https://custom.com/{0}'}

    # Test with non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with invalid YAML
    invalid_yaml_file = tmp_path / "invalid.yaml"
    with open(invalid_yaml_file, 'w') as f:
        f.write("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with non-dict YAML content
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    with open(non_dict_yaml_file, 'w') as f:
        f.write("- list item")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #12
#--------------------------

```python
def test_get_user_config():
    # Test default config
    config = get_user_config(default_config=True)
    assert config == DEFAULT_CONFIG

    # Test custom config dict
    custom_config = {'cookiecutters_dir': '/custom/dir'}
    config = get_user_config(default_config=custom_config)
    expected = merge_configs(DEFAULT_CONFIG, custom_config)
    assert config == expected

    # Test loading from custom config file
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("cookiecutters_dir: /test/dir\n")
        f.flush()
        config = get_user_config(config_file=f.name)
        assert config['cookiecutters_dir'] == '/test/dir'
        os.unlink(f.name)

    # Test loading from environment variable
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("replay_dir: /env/dir\n")
        f.flush()
        os.environ['COOKIECUTTER_CONFIG'] = f.name
        config = get_user_config()
        assert config['replay_dir'] == '/env/dir'
        del os.environ['COOKIECUTTER_CONFIG']
        os.unlink(f.name)

    # Test loading from default user config path
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("default_context:\n  key: value\n")
        f.flush()
        os.rename(f.name, USER_CONFIG_PATH)
        config = get_user_config()
        assert config['default_context'] == {'key': 'value'}
        os.unlink(USER_CONFIG_PATH)


# LLM-generated content at query #13
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "test_config.yaml"
    config_content = {
        'cookiecutters_dir': '~/test_cookiecutters/',
        'replay_dir': '~/test_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://custom.com/{0}'}
    }
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config_content, f)

    result = get_config(config_file)
    assert result['cookiecutters_dir'] == os.path.expanduser('~/test_cookiecutters/')
    assert result['replay_dir'] == os.path.expanduser('~/test_replay/')
    assert result['default_context'] == {'key': 'value'}
    assert result['abbreviations'] == {**BUILTIN_ABBREVIATIONS, 'custom': 'https://custom.com/{0}'}

    # Test with a non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with an invalid YAML file
    invalid_yaml_file = tmp_path / "invalid.yaml"
    with open(invalid_yaml_file, 'w', encoding='utf-8') as f:
        f.write("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with a YAML file that is not a dict
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    with open(non_dict_yaml_file, 'w', encoding='utf-8') as f:
        f.write("- list item 1\n- list item 2")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #14
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "test_config.yaml"
    config_content = {
        'cookiecutters_dir': '~/.test_cookiecutters/',
        'replay_dir': '~/.test_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://custom.com/{0}'}
    }
    with open(config_file, 'w') as f:
        yaml.dump(config_content, f)

    config = get_config(config_file)
    assert config['cookiecutters_dir'] == os.path.expanduser('~/.test_cookiecutters/')
    assert config['replay_dir'] == os.path.expanduser('~/.test_replay/')
    assert config['default_context'] == {'key': 'value'}
    assert config['abbreviations'] == {**BUILTIN_ABBREVIATIONS, 'custom': 'https://custom.com/{0}'}

    # Test with a non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with an invalid YAML file
    invalid_yaml_file = tmp_path / "invalid.yaml"
    with open(invalid_yaml_file, 'w') as f:
        f.write("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with a YAML file that is not a dict
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    with open(non_dict_yaml_file, 'w') as f:
        f.write("- list item")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #15
#--------------------------

```python
def test_get_config():
    # Test with a valid config file
    valid_config_content = """
cookiecutters_dir: /custom/cookiecutters/
replay_dir: /custom/replay/
default_context:
    key: value
abbreviations:
    custom: https://custom.com/{0}
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(valid_config_content)
        temp_config_path = f.name

    try:
        config = get_config(temp_config_path)
        assert config['cookiecutters_dir'] == '/custom/cookiecutters/'
        assert config['replay_dir'] == '/custom/replay/'
        assert config['default_context'] == {'key': 'value'}
        assert config['abbreviations']['custom'] == 'https://custom.com/{0}'
        assert config['abbreviations']['gh'] == 'https://github.com/{0}.git'  # Check default preserved
    finally:
        os.unlink(temp_config_path)

    # Test with non-existent config file
    with pytest.raises(ConfigDoesNotExistException):
        get_config('/non/existent/path.yaml')

    # Test with invalid YAML
    invalid_config_content = "invalid: yaml: content: [unclosed"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(invalid_config_content)
        temp_config_path = f.name

    try:
        with pytest.raises(InvalidConfiguration):
            get_config(temp_config_path)
    finally:
        os.unlink(temp_config_path)

    # Test with non-dict YAML content
    non_dict_config_content = "just a string"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(non_dict_config_content)
        temp_config_path = f.name

    try:
        with pytest.raises(InvalidConfiguration):
            get_config(temp_config_path)
    finally:
        os.unlink(temp_config_path)

    # Test path expansion
    config_with_env_vars = """
cookiecutters_dir: $HOME/test_cookiecutters/
replay_dir: ~/test_replay/
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_with_env_vars)
        temp_config_path = f.name

    try:
        config = get_config(temp_config_path)
        assert config['cookiecutters_dir'] == os.path.expanduser('~/test_cookiecutters/')
        assert config['replay_dir'] == os.path.expanduser('~/test_replay/')
    finally:
        os.unlink(temp_config_path)


# LLM-generated content at query #16
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text("""
cookiecutters_dir: /custom/cookiecutters/
replay_dir: /custom/replay/
default_context:
    key: value
abbreviations:
    custom: https://custom.com/{0}
""")

    result = get_config(config_file)
    assert result['cookiecutters_dir'] == '/custom/cookiecutters/'
    assert result['replay_dir'] == '/custom/replay/'
    assert result['default_context'] == {'key': 'value'}
    assert result['abbreviations']['custom'] == 'https://custom.com/{0}'
    assert result['abbreviations']['gh'] == 'https://github.com/{0}.git'

    # Test with environment variable expansion
    config_file.write_text("""
cookiecutters_dir: $HOME/test/
replay_dir: ~/replay/
""")
    result = get_config(config_file)
    assert result['cookiecutters_dir'] == os.path.expanduser('~') + '/test/'
    assert result['replay_dir'] == os.path.expanduser('~/replay/')

    # Test with non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with invalid YAML
    invalid_yaml_file = tmp_path / "invalid.yaml"
    invalid_yaml_file.write_text("invalid: yaml: content: [unclosed")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with non-dict YAML content
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    non_dict_yaml_file.write_text("- list item")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #17
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "test_config.yaml"
    config_content = {
        'cookiecutters_dir': '~/test_cookiecutters/',
        'replay_dir': '~/test_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://custom.com/{0}'}
    }
    with open(config_file, 'w') as f:
        yaml.dump(config_content, f)

    result = get_config(config_file)

    assert result['cookiecutters_dir'] == os.path.expanduser('~/test_cookiecutters/')
    assert result['replay_dir'] == os.path.expanduser('~/test_replay/')
    assert result['default_context'] == {'key': 'value'}
    assert result['abbreviations'] == {**BUILTIN_ABBREVIATIONS, 'custom': 'https://custom.com/{0}'}

    # Test with a non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with an invalid YAML file
    invalid_yaml_file = tmp_path / "invalid.yaml"
    with open(invalid_yaml_file, 'w') as f:
        f.write("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with a YAML file that is not a dict
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    with open(non_dict_yaml_file, 'w') as f:
        f.write("- list item")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #18
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "test_config.yaml"
    config_content = {
        'cookiecutters_dir': '~/test_cookiecutters/',
        'replay_dir': '~/test_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://custom.com/{0}'}
    }
    config_file.write_text(yaml.dump(config_content))

    result = get_config(config_file)
    assert result['cookiecutters_dir'] == os.path.expanduser('~/test_cookiecutters/')
    assert result['replay_dir'] == os.path.expanduser('~/test_replay/')
    assert result['default_context'] == {'key': 'value'}
    assert result['abbreviations'] == {**BUILTIN_ABBREVIATIONS, 'custom': 'https://custom.com/{0}'}

    # Test with a non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with an invalid YAML file
    invalid_yaml_file = tmp_path / "invalid.yaml"
    invalid_yaml_file.write_text("invalid: yaml: content: [unclosed")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with a YAML file that is not a dict
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    non_dict_yaml_file.write_text("- list item")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #19
#--------------------------

```python
def test_get_user_config():
    # Test default config
    result = get_user_config(default_config=True)
    assert result == DEFAULT_CONFIG

    # Test custom default config
    custom_default = {'cookiecutters_dir': '/custom/path'}
    result = get_user_config(default_config=custom_default)
    expected = merge_configs(DEFAULT_CONFIG, custom_default)
    assert result == expected

    # Test loading from custom config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump({'cookiecutters_dir': '/test/path'}, f)
        f.flush()
        result = get_user_config(config_file=f.name)
        expected = merge_configs(DEFAULT_CONFIG, {'cookiecutters_dir': '/test/path'})
        assert result == expected
        os.unlink(f.name)

    # Test loading from environment variable
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump({'replay_dir': '/env/path'}, f)
        f.flush()
        os.environ['COOKIECUTTER_CONFIG'] = f.name
        result = get_user_config()
        expected = merge_configs(DEFAULT_CONFIG, {'replay_dir': '/env/path'})
        assert result == expected
        del os.environ['COOKIECUTTER_CONFIG']
        os.unlink(f.name)

    # Test loading from default user config path
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump({'default_context': {'key': 'value'}}, f)
        f.flush()
        with patch('cookiecutter.config.USER_CONFIG_PATH', f.name):
            result = get_user_config()
            expected = merge_configs(DEFAULT_CONFIG, {'default_context': {'key': 'value'}})
            assert result == expected
        os.unlink(f.name)

    # Test non-existent config file raises exception
    with pytest.raises(ConfigDoesNotExistException):
        get_user_config(config_file='/non/existent/path.yaml')


# LLM-generated content at query #20
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "test_config.yaml"
    config_content = {
        'cookiecutters_dir': '~/test_cookiecutters/',
        'replay_dir': '~/test_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://custom.com/{0}'}
    }
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config_content, f)

    result = get_config(config_file)

    assert result['cookiecutters_dir'] == os.path.expanduser('~/test_cookiecutters/')
    assert result['replay_dir'] == os.path.expanduser('~/test_replay/')
    assert result['default_context'] == {'key': 'value'}
    assert result['abbreviations'] == {**BUILTIN_ABBREVIATIONS, 'custom': 'https://custom.com/{0}'}

    # Test with a non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with an invalid YAML file
    invalid_yaml_file = tmp_path / "invalid.yaml"
    with open(invalid_yaml_file, 'w', encoding='utf-8') as f:
        f.write("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with a YAML file that is not a dict
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    with open(non_dict_yaml_file, 'w', encoding='utf-8') as f:
        f.write("- list item")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #21
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "test_config.yaml"
    config_content = {
        'cookiecutters_dir': '~/test_cookiecutters/',
        'replay_dir': '~/test_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'test': 'https://test.com/{0}'}
    }
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config_content, f)

    result = get_config(config_file)
    assert result['cookiecutters_dir'] == os.path.expanduser('~/test_cookiecutters/')
    assert result['replay_dir'] == os.path.expanduser('~/test_replay/')
    assert result['default_context'] == {'key': 'value'}
    assert result['abbreviations'] == {**BUILTIN_ABBREVIATIONS, 'test': 'https://test.com/{0}'}

    # Test with a non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with an invalid YAML file
    invalid_yaml_file = tmp_path / "invalid.yaml"
    with open(invalid_yaml_file, 'w', encoding='utf-8') as f:
        f.write("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with a YAML file that is not a dict
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    with open(non_dict_yaml_file, 'w', encoding='utf-8') as f:
        f.write("- not a dict")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #22
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "test_config.yaml"
    config_content = {
        'cookiecutters_dir': '~/test_cookiecutters/',
        'replay_dir': '~/test_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://custom.com/{0}'}
    }
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config_content, f)

    config = get_config(config_file)
    assert config['cookiecutters_dir'] == os.path.expanduser('~/test_cookiecutters/')
    assert config['replay_dir'] == os.path.expanduser('~/test_replay/')
    assert config['default_context'] == {'key': 'value'}
    assert config['abbreviations'] == {**BUILTIN_ABBREVIATIONS, 'custom': 'https://custom.com/{0}'}

    # Test with a non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with an invalid YAML file
    invalid_yaml_file = tmp_path / "invalid.yaml"
    with open(invalid_yaml_file, 'w', encoding='utf-8') as f:
        f.write("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with a YAML file that is not a dict
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    with open(non_dict_yaml_file, 'w', encoding='utf-8') as f:
        f.write("- list item 1\n- list item 2")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #23
#--------------------------

```python
def test_get_config():
    # Test with a valid config file
    valid_config = {
        'cookiecutters_dir': '~/test_cookiecutters/',
        'replay_dir': '~/test_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://custom.com/{0}'}
    }
    config_file = 'valid_config.yaml'
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(valid_config, f)

    config = get_config(config_file)
    assert config['cookiecutters_dir'] == os.path.expanduser('~/test_cookiecutters/')
    assert config['replay_dir'] == os.path.expanduser('~/test_replay/')
    assert config['default_context'] == {'key': 'value'}
    assert config['abbreviations'] == {**BUILTIN_ABBREVIATIONS, 'custom': 'https://custom.com/{0}'}

    os.remove(config_file)

    # Test with a non-existent config file
    with pytest.raises(ConfigDoesNotExistException):
        get_config('non_existent_config.yaml')

    # Test with an invalid YAML file
    invalid_config_file = 'invalid_config.yaml'
    with open(invalid_config_file, 'w', encoding='utf-8') as f:
        f.write('invalid: yaml: content: [')

    with pytest.raises(InvalidConfiguration):
        get_config(invalid_config_file)

    os.remove(invalid_config_file)

    # Test with a YAML file that is not a dict
    non_dict_config_file = 'non_dict_config.yaml'
    with open(non_dict_config_file, 'w', encoding='utf-8') as f:
        f.write('- not a dict')

    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_config_file)

    os.remove(non_dict_config_file)


# LLM-generated content at query #24
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(
        """
cookiecutters_dir: /custom/cookiecutters/
replay_dir: /custom/replay/
default_context:
    key1: value1
abbreviations:
    custom: https://custom.com/{0}
"""
    )

    result = get_config(config_file)
    assert result["cookiecutters_dir"] == "/custom/cookiecutters/"
    assert result["replay_dir"] == "/custom/replay/"
    assert result["default_context"] == {"key1": "value1"}
    assert result["abbreviations"]["custom"] == "https://custom.com/{0}"
    assert result["abbreviations"]["gh"] == "https://github.com/{0}.git"

    # Test with environment variable expansion
    config_file.write_text(
        """
cookiecutters_dir: $HOME/test_cookiecutters/
replay_dir: ~/test_replay/
"""
    )
    result = get_config(config_file)
    assert result["cookiecutters_dir"] == os.path.expandvars("$HOME/test_cookiecutters/")
    assert result["replay_dir"] == os.path.expanduser("~/test_replay/")

    # Test with non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with invalid YAML
    invalid_yaml_file = tmp_path / "invalid.yaml"
    invalid_yaml_file.write_text("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with non-dict YAML content
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    non_dict_yaml_file.write_text("- list item")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #25
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "test_config.yaml"
    config_content = {
        'cookiecutters_dir': '~/test_cookiecutters/',
        'replay_dir': '~/test_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://custom.com/{0}'}
    }
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config_content, f)

    result = get_config(config_file)
    assert result['cookiecutters_dir'] == os.path.expanduser('~/test_cookiecutters/')
    assert result['replay_dir'] == os.path.expanduser('~/test_replay/')
    assert result['default_context'] == {'key': 'value'}
    assert result['abbreviations'] == {**BUILTIN_ABBREVIATIONS, 'custom': 'https://custom.com/{0}'}

    # Test with a non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with an invalid YAML file
    invalid_yaml_file = tmp_path / "invalid.yaml"
    with open(invalid_yaml_file, 'w', encoding='utf-8') as f:
        f.write("invalid: yaml: content: [unclosed")

    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with a YAML file that is not a dict
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    with open(non_dict_yaml_file, 'w', encoding='utf-8') as f:
        f.write("- list item 1\n- list item 2")

    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #26
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "config.yaml"
    config_file.write_text("""
cookiecutters_dir: /custom/cookiecutters/
replay_dir: /custom/replay/
default_context:
    key: value
abbreviations:
    custom: https://custom.com/{0}
""")

    config = get_config(config_file)
    assert config['cookiecutters_dir'] == '/custom/cookiecutters/'
    assert config['replay_dir'] == '/custom/replay/'
    assert config['default_context'] == {'key': 'value'}
    assert config['abbreviations']['custom'] == 'https://custom.com/{0}'
    assert config['abbreviations']['gh'] == 'https://github.com/{0}.git'

    # Test with environment variable expansion
    config_file.write_text("""
cookiecutters_dir: $HOME/test/
replay_dir: ~/replay/
""")
    config = get_config(config_file)
    assert config['cookiecutters_dir'] == os.path.expandvars('$HOME/test/')
    assert config['replay_dir'] == os.path.expanduser('~/replay/')

    # Test with non-existent config file
    with pytest.raises(ConfigDoesNotExistException):
        get_config(tmp_path / "nonexistent.yaml")

    # Test with invalid YAML
    config_file.write_text("invalid yaml content")
    with pytest.raises(InvalidConfiguration):
        get_config(config_file)

    # Test with non-dict YAML content
    config_file.write_text("- list item")
    with pytest.raises(InvalidConfiguration):
        get_config(config_file)


# LLM-generated content at query #27
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "test_config.yaml"
    config_content = {
        'cookiecutters_dir': '~/test_cookiecutters/',
        'replay_dir': '~/test_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://custom.com/{0}'}
    }
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config_content, f)

    config = get_config(config_file)
    assert config['cookiecutters_dir'] == os.path.expanduser('~/test_cookiecutters/')
    assert config['replay_dir'] == os.path.expanduser('~/test_replay/')
    assert config['default_context'] == {'key': 'value'}
    assert config['abbreviations'] == {**BUILTIN_ABBREVIATIONS, 'custom': 'https://custom.com/{0}'}

    # Test with a non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with an invalid YAML file
    invalid_yaml_file = tmp_path / "invalid.yaml"
    with open(invalid_yaml_file, 'w', encoding='utf-8') as f:
        f.write("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with a YAML file that is not a dict
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    with open(non_dict_yaml_file, 'w', encoding='utf-8') as f:
        f.write("- not a dict")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #28
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "test_config.yaml"
    config_content = {
        'cookiecutters_dir': '~/test_cookiecutters/',
        'replay_dir': '~/test_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://custom.com/{0}'}
    }
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config_content, f)

    result = get_config(config_file)

    assert result['cookiecutters_dir'] == os.path.expanduser('~/test_cookiecutters/')
    assert result['replay_dir'] == os.path.expanduser('~/test_replay/')
    assert result['default_context'] == {'key': 'value'}
    assert result['abbreviations'] == {**BUILTIN_ABBREVIATIONS, 'custom': 'https://custom.com/{0}'}
    assert isinstance(result['default_context'], collections.OrderedDict)

    # Test with a non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with an invalid YAML file
    invalid_yaml_file = tmp_path / "invalid.yaml"
    with open(invalid_yaml_file, 'w', encoding='utf-8') as f:
        f.write("invalid: yaml: content: [")

    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with a YAML file that is not a dict
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    with open(non_dict_yaml_file, 'w', encoding='utf-8') as f:
        f.write("- list item 1\n- list item 2")

    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #29
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "test_config.yaml"
    config_content = {
        'cookiecutters_dir': '~/test_cookiecutters/',
        'replay_dir': '~/test_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://custom.com/{0}'}
    }
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config_content, f)

    result = get_config(config_file)
    assert result['cookiecutters_dir'] == os.path.expanduser('~/test_cookiecutters/')
    assert result['replay_dir'] == os.path.expanduser('~/test_replay/')
    assert result['default_context'] == {'key': 'value'}
    assert result['abbreviations'] == {**BUILTIN_ABBREVIATIONS, **{'custom': 'https://custom.com/{0}'}}

    # Test with a non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with an invalid YAML file
    invalid_yaml_file = tmp_path / "invalid.yaml"
    with open(invalid_yaml_file, 'w', encoding='utf-8') as f:
        f.write("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with a YAML file that is not a dict
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    with open(non_dict_yaml_file, 'w', encoding='utf-8') as f:
        f.write("- list_item1\n- list_item2")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #30
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / 'config.yaml'
    config_file.write_text('''
cookiecutters_dir: ~/test_dir/
replay_dir: ~/test_replay/
default_context:
    key1: value1
abbreviations:
    custom: https://custom.com/{0}
''')

    config = get_config(config_file)
    assert config['cookiecutters_dir'] == os.path.expanduser('~/test_dir/')
    assert config['replay_dir'] == os.path.expanduser('~/test_replay/')
    assert config['default_context'] == {'key1': 'value1'}
    assert config['abbreviations']['custom'] == 'https://custom.com/{0}'
    assert config['abbreviations']['gh'] == 'https://github.com/{0}.git'

    # Test with a non-existent config file
    non_existent_file = tmp_path / 'non_existent.yaml'
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with an invalid YAML file
    invalid_yaml_file = tmp_path / 'invalid.yaml'
    invalid_yaml_file.write_text('invalid: yaml: content: [')
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with a YAML file that is not a dict
    non_dict_yaml_file = tmp_path / 'non_dict.yaml'
    non_dict_yaml_file.write_text('- list item')
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #31
#--------------------------

```python
def test_get_user_config():
    # Test default config
    config = get_user_config(default_config=True)
    assert config == DEFAULT_CONFIG

    # Test custom config dict
    custom_config = {'cookiecutters_dir': '/custom/dir'}
    config = get_user_config(default_config=custom_config)
    expected = merge_configs(DEFAULT_CONFIG, custom_config)
    assert config == expected

    # Test loading from custom config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump({'cookiecutters_dir': '/test/dir'}, f)
        config = get_user_config(config_file=f.name)
        assert config['cookiecutters_dir'] == '/test/dir'
        os.unlink(f.name)

    # Test loading from environment variable
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump({'replay_dir': '/env/dir'}, f)
        os.environ['COOKIECUTTER_CONFIG'] = f.name
        config = get_user_config()
        assert config['replay_dir'] == '/env/dir'
        del os.environ['COOKIECUTTER_CONFIG']
        os.unlink(f.name)

    # Test loading from default user config path
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump({'default_context': {'key': 'value'}}, f)
        original_user_config = USER_CONFIG_PATH
        USER_CONFIG_PATH = f.name
        config = get_user_config()
        assert config['default_context'] == {'key': 'value'}
        USER_CONFIG_PATH = original_user_config
        os.unlink(f.name)


# LLM-generated content at query #32
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "test_config.yaml"
    config_content = {
        'cookiecutters_dir': '~/test_cookiecutters/',
        'replay_dir': '~/test_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://custom.com/{0}'}
    }
    with open(config_file, 'w') as f:
        yaml.dump(config_content, f)

    result = get_config(config_file)
    assert result['cookiecutters_dir'] == os.path.expanduser('~/test_cookiecutters/')
    assert result['replay_dir'] == os.path.expanduser('~/test_replay/')
    assert result['default_context'] == {'key': 'value'}
    assert result['abbreviations'] == {**BUILTIN_ABBREVIATIONS, 'custom': 'https://custom.com/{0}'}

    # Test with a non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with an invalid YAML file
    invalid_file = tmp_path / "invalid.yaml"
    with open(invalid_file, 'w') as f:
        f.write("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_file)

    # Test with a YAML file that is not a dict
    non_dict_file = tmp_path / "non_dict.yaml"
    with open(non_dict_file, 'w') as f:
        f.write("- not a dict")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_file)


# LLM-generated content at query #33
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "test_config.yaml"
    config_content = {
        'cookiecutters_dir': '~/test_cookiecutters/',
        'replay_dir': '~/test_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://custom.com/{0}'}
    }
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config_content, f)

    result = get_config(config_file)
    assert result['cookiecutters_dir'] == os.path.expanduser('~/test_cookiecutters/')
    assert result['replay_dir'] == os.path.expanduser('~/test_replay/')
    assert result['default_context'] == {'key': 'value'}
    assert result['abbreviations'] == {**BUILTIN_ABBREVIATIONS, 'custom': 'https://custom.com/{0}'}

    # Test with a non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with an invalid YAML file
    invalid_yaml_file = tmp_path / "invalid.yaml"
    with open(invalid_yaml_file, 'w', encoding='utf-8') as f:
        f.write("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with a YAML file that is not a dict
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    with open(non_dict_yaml_file, 'w', encoding='utf-8') as f:
        f.write("- list item")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #34
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "test_config.yaml"
    config_content = {
        'cookiecutters_dir': '~/test_cookiecutters/',
        'replay_dir': '~/test_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://custom.com/{0}'}
    }
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config_content, f)

    result = get_config(config_file)
    assert result['cookiecutters_dir'] == os.path.expanduser('~/test_cookiecutters/')
    assert result['replay_dir'] == os.path.expanduser('~/test_replay/')
    assert result['default_context'] == {'key': 'value'}
    assert result['abbreviations'] == {**BUILTIN_ABBREVIATIONS, 'custom': 'https://custom.com/{0}'}

    # Test with a non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with an invalid YAML file
    invalid_yaml_file = tmp_path / "invalid.yaml"
    with open(invalid_yaml_file, 'w', encoding='utf-8') as f:
        f.write("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with a YAML file that is not a dict
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    with open(non_dict_yaml_file, 'w', encoding='utf-8') as f:
        f.write("- list item")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #35
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "test_config.yaml"
    config_content = {
        'cookiecutters_dir': '~/test_cookiecutters/',
        'replay_dir': '~/test_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://custom.com/{0}'}
    }
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config_content, f)

    config = get_config(config_file)
    assert config['cookiecutters_dir'] == os.path.expanduser('~/test_cookiecutters/')
    assert config['replay_dir'] == os.path.expanduser('~/test_replay/')
    assert config['default_context'] == {'key': 'value'}
    assert config['abbreviations'] == {**BUILTIN_ABBREVIATIONS, 'custom': 'https://custom.com/{0}'}

    # Test with a non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with an invalid YAML file
    invalid_yaml_file = tmp_path / "invalid.yaml"
    with open(invalid_yaml_file, 'w', encoding='utf-8') as f:
        f.write("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with a YAML file that is not a dict
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    with open(non_dict_yaml_file, 'w', encoding='utf-8') as f:
        f.write("- list item")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #36
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "config.yaml"
    config_content = {
        'cookiecutters_dir': '~/test_cookiecutters/',
        'replay_dir': '~/test_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://custom.com/{0}'}
    }
    with open(config_file, 'w') as f:
        yaml.dump(config_content, f)

    result = get_config(config_file)

    assert result['cookiecutters_dir'] == os.path.expanduser('~/test_cookiecutters/')
    assert result['replay_dir'] == os.path.expanduser('~/test_replay/')
    assert result['default_context'] == {'key': 'value'}
    assert result['abbreviations'] == {**BUILTIN_ABBREVIATIONS, 'custom': 'https://custom.com/{0}'}
    assert isinstance(result['default_context'], collections.OrderedDict)

    # Test with a non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with an invalid YAML file
    invalid_yaml_file = tmp_path / "invalid.yaml"
    with open(invalid_yaml_file, 'w') as f:
        f.write("invalid: yaml: content: [unclosed")

    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with a YAML file that is not a dict
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    with open(non_dict_yaml_file, 'w') as f:
        f.write("- list item 1\n- list item 2")

    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #37
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / 'cookiecutterrc'
    config_content = {
        'cookiecutters_dir': '~/test_cookiecutters/',
        'replay_dir': '~/test_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://custom.com/{0}'}
    }
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config_content, f)

    result = get_config(config_file)
    assert result['cookiecutters_dir'] == os.path.expanduser('~/test_cookiecutters/')
    assert result['replay_dir'] == os.path.expanduser('~/test_replay/')
    assert result['default_context'] == {'key': 'value'}
    assert result['abbreviations'] == {**BUILTIN_ABBREVIATIONS, 'custom': 'https://custom.com/{0}'}

    # Test with a non-existent config file
    non_existent_file = tmp_path / 'non_existent_config'
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with an invalid YAML file
    invalid_yaml_file = tmp_path / 'invalid_yaml'
    with open(invalid_yaml_file, 'w', encoding='utf-8') as f:
        f.write('invalid: yaml: content: [')
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with a YAML file that is not a dict
    non_dict_yaml_file = tmp_path / 'non_dict_yaml'
    with open(non_dict_yaml_file, 'w', encoding='utf-8') as f:
        f.write('- list item 1\n- list item 2')
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #38
#--------------------------

```python
def test_get_config():
    # Test with a valid config file
    valid_config = {
        'cookiecutters_dir': '~/test_dir/',
        'replay_dir': '~/test_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'test': 'test_url'}
    }
    config_path = 'test_config.yaml'
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(valid_config, f)

    result = get_config(config_path)
    assert result['cookiecutters_dir'] == os.path.expanduser('~/test_dir/')
    assert result['replay_dir'] == os.path.expanduser('~/test_replay/')
    assert result['default_context'] == {'key': 'value'}
    assert result['abbreviations'] == {**BUILTIN_ABBREVIATIONS, 'test': 'test_url'}
    os.remove(config_path)

    # Test with a non-existent config file
    with pytest.raises(ConfigDoesNotExistException):
        get_config('non_existent_config.yaml')

    # Test with an invalid YAML file
    with open('invalid_config.yaml', 'w', encoding='utf-8') as f:
        f.write('invalid: yaml: content: [')
    with pytest.raises(InvalidConfiguration):
        get_config('invalid_config.yaml')
    os.remove('invalid_config.yaml')

    # Test with a YAML file that is not a dict
    with open('non_dict_config.yaml', 'w', encoding='utf-8') as f:
        f.write('- not a dict')
    with pytest.raises(InvalidConfiguration):
        get_config('non_dict_config.yaml')
    os.remove('non_dict_config.yaml')


# LLM-generated content at query #39
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "test_config.yaml"
    config_content = {
        'cookiecutters_dir': '~/test_cookiecutters/',
        'replay_dir': '~/test_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://custom.com/{0}'}
    }
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config_content, f)

    result = get_config(config_file)
    assert result['cookiecutters_dir'] == os.path.expanduser('~/test_cookiecutters/')
    assert result['replay_dir'] == os.path.expanduser('~/test_replay/')
    assert result['default_context'] == {'key': 'value'}
    assert result['abbreviations'] == {**BUILTIN_ABBREVIATIONS, 'custom': 'https://custom.com/{0}'}

    # Test with a non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with an invalid YAML file
    invalid_yaml_file = tmp_path / "invalid.yaml"
    with open(invalid_yaml_file, 'w', encoding='utf-8') as f:
        f.write("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with a YAML file that is not a dict
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    with open(non_dict_yaml_file, 'w', encoding='utf-8') as f:
        f.write("- list item")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #40
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "config.yaml"
    config_file.write_text("""
cookiecutters_dir: /custom/cookiecutters/
replay_dir: /custom/replay/
default_context:
    key: value
abbreviations:
    custom: https://custom.com/{0}
""")

    config = get_config(config_file)
    assert config['cookiecutters_dir'] == '/custom/cookiecutters/'
    assert config['replay_dir'] == '/custom/replay/'
    assert config['default_context'] == {'key': 'value'}
    assert config['abbreviations']['custom'] == 'https://custom.com/{0}'
    assert config['abbreviations']['gh'] == 'https://github.com/{0}.git'

    # Test with environment variable expansion
    config_file.write_text("""
cookiecutters_dir: $HOME/test/
replay_dir: ~/test/
""")
    config = get_config(config_file)
    assert config['cookiecutters_dir'] == os.path.expandvars('$HOME/test/')
    assert config['replay_dir'] == os.path.expanduser('~/test/')

    # Test with non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with invalid YAML
    invalid_yaml_file = tmp_path / "invalid.yaml"
    invalid_yaml_file.write_text("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with non-dict YAML content
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    non_dict_yaml_file.write_text("this is not a dict")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #41
#--------------------------

```python
def test_get_config(tmp_path):
    # Test successful config loading
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text("""
cookiecutters_dir: /custom/cookiecutters/
replay_dir: /custom/replay/
default_context:
    key: value
abbreviations:
    custom: https://custom.com/{0}
""")

    result = get_config(config_file)
    assert result['cookiecutters_dir'] == '/custom/cookiecutters/'
    assert result['replay_dir'] == '/custom/replay/'
    assert result['default_context'] == {'key': 'value'}
    assert result['abbreviations']['custom'] == 'https://custom.com/{0}'
    assert result['abbreviations']['gh'] == 'https://github.com/{0}.git'

    # Test path expansion
    config_file.write_text("""
cookiecutters_dir: ~/expanded/cookiecutters/
replay_dir: ~/expanded/replay/
""")
    result = get_config(config_file)
    assert result['cookiecutters_dir'] == os.path.expanduser('~/expanded/cookiecutters/')
    assert result['replay_dir'] == os.path.expanduser('~/expanded/replay/')

    # Test file not found
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test invalid YAML
    invalid_yaml_file = tmp_path / "invalid.yaml"
    invalid_yaml_file.write_text("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test non-dict YAML content
    non_dict_file = tmp_path / "non_dict.yaml"
    non_dict_file.write_text("- list item")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_file)


# LLM-generated content at query #42
#--------------------------

```python
def test_get_user_config():
    # Test with default_config=True
    config = get_user_config(default_config=True)
    assert config == DEFAULT_CONFIG

    # Test with default_config as a dict
    custom_config = {'cookiecutters_dir': '/custom/path'}
    config = get_user_config(default_config=custom_config)
    expected_config = merge_configs(DEFAULT_CONFIG, custom_config)
    assert config == expected_config

    # Test with a custom config file
    with patch('builtins.open', mock_open(read_data='cookiecutters_dir: /custom/path')):
        with patch('os.path.exists', return_value=True):
            config = get_user_config(config_file='/custom/config')
            assert config['cookiecutters_dir'] == '/custom/path'

    # Test with COOKIECUTTER_CONFIG environment variable
    with patch.dict('os.environ', {'COOKIECUTTER_CONFIG': '/env/config'}):
        with patch('builtins.open', mock_open(read_data='cookiecutters_dir: /env/path')):
            config = get_user_config()
            assert config['cookiecutters_dir'] == '/env/path'

    # Test with default config file path
    with patch('os.path.exists', return_value=True):
        with patch('builtins.open', mock_open(read_data='cookiecutters_dir: /default/path')):
            config = get_user_config()
            assert config['cookiecutters_dir'] == '/default/path'

    # Test with non-existent config file
    with patch('os.path.exists', return_value=False):
        config = get_user_config()
        assert config == DEFAULT_CONFIG


# LLM-generated content at query #43
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "test_config.yaml"
    config_content = {
        'cookiecutters_dir': '~/test_cookiecutters/',
        'replay_dir': '~/test_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://custom.com/{0}'}
    }
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config_content, f)

    config = get_config(config_file)
    assert config['cookiecutters_dir'] == os.path.expanduser('~/test_cookiecutters/')
    assert config['replay_dir'] == os.path.expanduser('~/test_replay/')
    assert config['default_context'] == {'key': 'value'}
    assert config['abbreviations'] == {**BUILTIN_ABBREVIATIONS, 'custom': 'https://custom.com/{0}'}

    # Test with non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with invalid YAML
    invalid_yaml_file = tmp_path / "invalid.yaml"
    with open(invalid_yaml_file, 'w', encoding='utf-8') as f:
        f.write("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with non-dict YAML content
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    with open(non_dict_yaml_file, 'w', encoding='utf-8') as f:
        f.write("- list item")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #44
#--------------------------

```python
def test_get_user_config():
    # Test default config
    config = get_user_config(default_config=True)
    assert config == DEFAULT_CONFIG

    # Test custom config dict
    custom_config = {'cookiecutters_dir': '/custom/dir'}
    config = get_user_config(default_config=custom_config)
    expected = merge_configs(DEFAULT_CONFIG, custom_config)
    assert config == expected

    # Test config file path
    with pytest.raises(ConfigDoesNotExistException):
        get_user_config(config_file='/non/existent/path')

    # Test environment variable
    with pytest.raises(ConfigDoesNotExistException):
        os.environ['COOKIECUTTER_CONFIG'] = '/non/existent/path'
        get_user_config()

    # Test default config file path
    with patch('os.path.exists', return_value=True):
        with patch('builtins.open', mock_open(read_data='cookiecutters_dir: /test/dir')):
            config = get_user_config()
            assert config['cookiecutters_dir'] == '/test/dir'

    # Test non-existent default config file path
    with patch('os.path.exists', return_value=False):
        config = get_user_config()
        assert config == DEFAULT_CONFIG


# LLM-generated content at query #45
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "test_config.yaml"
    config_content = {
        'cookiecutters_dir': '~/test_cookiecutters/',
        'replay_dir': '~/test_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://custom.com/{0}'}
    }
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config_content, f)

    result = get_config(config_file)
    assert result['cookiecutters_dir'] == os.path.expanduser('~/test_cookiecutters/')
    assert result['replay_dir'] == os.path.expanduser('~/test_replay/')
    assert result['default_context'] == {'key': 'value'}
    assert result['abbreviations'] == {**BUILTIN_ABBREVIATIONS, 'custom': 'https://custom.com/{0}'}

    # Test with a non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with an invalid YAML file
    invalid_yaml_file = tmp_path / "invalid.yaml"
    with open(invalid_yaml_file, 'w', encoding='utf-8') as f:
        f.write("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with a YAML file that is not a dict
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    with open(non_dict_yaml_file, 'w', encoding='utf-8') as f:
        f.write("- list item 1\n- list item 2")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #46
#--------------------------

```python
def test_get_user_config():
    # Test default config
    config = get_user_config(default_config=True)
    assert config == DEFAULT_CONFIG

    # Test custom config dict
    custom_config = {'cookiecutters_dir': '/custom/dir'}
    config = get_user_config(default_config=custom_config)
    expected = merge_configs(DEFAULT_CONFIG, custom_config)
    assert config == expected

    # Test loading from custom config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump({'cookiecutters_dir': '/test/dir'}, f)
        config = get_user_config(config_file=f.name)
        assert config['cookiecutters_dir'] == '/test/dir'
        os.unlink(f.name)

    # Test loading from environment variable
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump({'replay_dir': '/env/dir'}, f)
        os.environ['COOKIECUTTER_CONFIG'] = f.name
        config = get_user_config()
        assert config['replay_dir'] == '/env/dir'
        del os.environ['COOKIECUTTER_CONFIG']
        os.unlink(f.name)

    # Test loading from default user config path
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump({'default_context': {'key': 'value'}}, f)
        original_path = USER_CONFIG_PATH
        USER_CONFIG_PATH = f.name
        config = get_user_config()
        assert config['default_context'] == {'key': 'value'}
        USER_CONFIG_PATH = original_path
        os.unlink(f.name)

    # Test non-existent config file raises exception
    with pytest.raises(ConfigDoesNotExistException):
        get_user_config(config_file='/nonexistent/path.yaml')


# LLM-generated content at query #47
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "test_config.yaml"
    config_content = {
        'cookiecutters_dir': '~/custom_cookiecutters/',
        'replay_dir': '~/custom_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://custom.com/{0}'}
    }
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config_content, f)

    config = get_config(config_file)
    assert config['cookiecutters_dir'] == os.path.expanduser('~/custom_cookiecutters/')
    assert config['replay_dir'] == os.path.expanduser('~/custom_replay/')
    assert config['default_context'] == {'key': 'value'}
    assert config['abbreviations'] == {**BUILTIN_ABBREVIATIONS, 'custom': 'https://custom.com/{0}'}

    # Test with a non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with an invalid YAML file
    invalid_yaml_file = tmp_path / "invalid.yaml"
    with open(invalid_yaml_file, 'w', encoding='utf-8') as f:
        f.write("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with a YAML file that is not a dict
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    with open(non_dict_yaml_file, 'w', encoding='utf-8') as f:
        f.write("- list item")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #48
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
        cookiecutters_dir: /custom/cookiecutters/
        replay_dir: /custom/replay/
        default_context:
            key1: value1
        abbreviations:
            custom: https://custom.com/{0}
        """
    )

    config = get_config(config_file)
    assert config["cookiecutters_dir"] == "/custom/cookiecutters/"
    assert config["replay_dir"] == "/custom/replay/"
    assert config["default_context"] == {"key1": "value1"}
    assert config["abbreviations"]["custom"] == "https://custom.com/{0}"
    assert config["abbreviations"]["gh"] == "https://github.com/{0}.git"

    # Test with environment variable expansion
    config_file.write_text(
        """
        cookiecutters_dir: $HOME/test/
        replay_dir: ~/replay/
        """
    )
    config = get_config(config_file)
    assert config["cookiecutters_dir"] == os.path.expandvars("$HOME/test/")
    assert config["replay_dir"] == os.path.expanduser("~/replay/")

    # Test with non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with invalid YAML
    invalid_yaml_file = tmp_path / "invalid.yaml"
    invalid_yaml_file.write_text("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with non-dict YAML content
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    non_dict_yaml_file.write_text("- list item")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #49
#--------------------------

```python
def test_get_user_config():
    # Test default config
    config = get_user_config(default_config=True)
    assert config == DEFAULT_CONFIG

    # Test custom default config
    custom_default = {'cookiecutters_dir': '/custom/dir'}
    config = get_user_config(default_config=custom_default)
    expected = merge_configs(DEFAULT_CONFIG, custom_default)
    assert config == expected

    # Test loading from custom config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write('cookiecutters_dir: /test/dir\nreplay_dir: /test/replay')
        f.flush()
        config = get_user_config(config_file=f.name)
        assert config['cookiecutters_dir'] == '/test/dir'
        assert config['replay_dir'] == '/test/replay'
        os.unlink(f.name)

    # Test loading from environment variable
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write('cookiecutters_dir: /env/dir')
        f.flush()
        os.environ['COOKIECUTTER_CONFIG'] = f.name
        config = get_user_config()
        assert config['cookiecutters_dir'] == '/env/dir'
        del os.environ['COOKIECUTTER_CONFIG']
        os.unlink(f.name)

    # Test loading from default user config path
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write('cookiecutters_dir: /user/dir')
        f.flush()
        os.rename(f.name, USER_CONFIG_PATH)
        config = get_user_config()
        assert config['cookiecutters_dir'] == '/user/dir'
        os.unlink(USER_CONFIG_PATH)

    # Test fallback to default config when no config file exists
    if os.path.exists(USER_CONFIG_PATH):
        os.unlink(USER_CONFIG_PATH)
    config = get_user_config()
    assert config == DEFAULT_CONFIG


# LLM-generated content at query #50
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "test_config.yaml"
    config_content = {
        'cookiecutters_dir': '~/test_cookiecutters/',
        'replay_dir': '~/test_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://custom.com/{0}'}
    }
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config_content, f)

    result = get_config(config_file)
    assert result['cookiecutters_dir'] == os.path.expanduser('~/test_cookiecutters/')
    assert result['replay_dir'] == os.path.expanduser('~/test_replay/')
    assert result['default_context'] == {'key': 'value'}
    assert result['abbreviations'] == {**BUILTIN_ABBREVIATIONS, **{'custom': 'https://custom.com/{0}'}}

    # Test with a non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with an invalid YAML file
    invalid_yaml_file = tmp_path / "invalid.yaml"
    with open(invalid_yaml_file, 'w', encoding='utf-8') as f:
        f.write("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with a YAML file that is not a dict
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    with open(non_dict_yaml_file, 'w', encoding='utf-8') as f:
        f.write("- list item 1\n- list item 2")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #51
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / 'config.yaml'
    config_content = {
        'cookiecutters_dir': '~/test_cookiecutters/',
        'replay_dir': '~/test_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://custom.com/{0}'}
    }
    with open(config_file, 'w') as f:
        yaml.dump(config_content, f)

    result = get_config(config_file)
    assert result['cookiecutters_dir'] == os.path.expanduser('~/test_cookiecutters/')
    assert result['replay_dir'] == os.path.expanduser('~/test_replay/')
    assert result['default_context'] == {'key': 'value'}
    assert result['abbreviations'] == {'gh': 'https://github.com/{0}.git', 'gl': 'https://gitlab.com/{0}.git', 'bb': 'https://bitbucket.org/{0}', 'custom': 'https://custom.com/{0}'}

    # Test with a non-existent config file
    non_existent_file = tmp_path / 'non_existent.yaml'
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with an invalid YAML file
    invalid_yaml_file = tmp_path / 'invalid.yaml'
    with open(invalid_yaml_file, 'w') as f:
        f.write("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with a YAML file that is not a dict
    non_dict_yaml_file = tmp_path / 'non_dict.yaml'
    with open(non_dict_yaml_file, 'w') as f:
        f.write("- not a dict")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #52
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "config.yaml"
    config_content = {
        'cookiecutters_dir': '~/test_cookiecutters/',
        'replay_dir': '~/test_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://custom.com/{0}'}
    }
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config_content, f)

    result = get_config(config_file)

    assert result['cookiecutters_dir'] == os.path.expanduser('~/test_cookiecutters/')
    assert result['replay_dir'] == os.path.expanduser('~/test_replay/')
    assert result['default_context'] == {'key': 'value'}
    assert result['abbreviations'] == {**BUILTIN_ABBREVIATIONS, 'custom': 'https://custom.com/{0}'}

    # Test with a non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with an invalid YAML file
    invalid_yaml_file = tmp_path / "invalid.yaml"
    with open(invalid_yaml_file, 'w', encoding='utf-8') as f:
        f.write("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with a YAML file that is not a dict
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    with open(non_dict_yaml_file, 'w', encoding='utf-8') as f:
        f.write("- list item 1\n- list item 2")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #53
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "test_config.yaml"
    config_content = {
        'cookiecutters_dir': '~/test_cookiecutters/',
        'replay_dir': '~/test_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://custom.com/{0}'}
    }
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config_content, f)

    config = get_config(config_file)
    assert config['cookiecutters_dir'] == os.path.expanduser('~/test_cookiecutters/')
    assert config['replay_dir'] == os.path.expanduser('~/test_replay/')
    assert config['default_context'] == {'key': 'value'}
    assert config['abbreviations']['custom'] == 'https://custom.com/{0}'
    assert config['abbreviations']['gh'] == 'https://github.com/{0}.git'

    # Test with a non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with an invalid YAML file
    invalid_yaml_file = tmp_path / "invalid.yaml"
    with open(invalid_yaml_file, 'w', encoding='utf-8') as f:
        f.write("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with a YAML file that is not a dict
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    with open(non_dict_yaml_file, 'w', encoding='utf-8') as f:
        f.write("- list_item1\n- list_item2")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #54
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "test_config.yaml"
    config_content = {
        'cookiecutters_dir': '~/test_cookiecutters/',
        'replay_dir': '~/test_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://custom.com/{0}'}
    }
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config_content, f)

    result = get_config(config_file)
    assert result['cookiecutters_dir'] == os.path.expanduser('~/test_cookiecutters/')
    assert result['replay_dir'] == os.path.expanduser('~/test_replay/')
    assert result['default_context'] == {'key': 'value'}
    assert result['abbreviations'] == {**BUILTIN_ABBREVIATIONS, 'custom': 'https://custom.com/{0}'}

    # Test with a non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with an invalid YAML file
    invalid_yaml_file = tmp_path / "invalid.yaml"
    with open(invalid_yaml_file, 'w', encoding='utf-8') as f:
        f.write("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with a YAML file that is not a dict
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    with open(non_dict_yaml_file, 'w', encoding='utf-8') as f:
        f.write("- list_item1\n- list_item2")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #55
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "test_config.yaml"
    config_content = {
        'cookiecutters_dir': '~/test_cookiecutters/',
        'replay_dir': '~/test_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://custom.com/{0}'}
    }
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config_content, f)

    config = get_config(config_file)
    assert config['cookiecutters_dir'] == os.path.expanduser('~/test_cookiecutters/')
    assert config['replay_dir'] == os.path.expanduser('~/test_replay/')
    assert config['default_context'] == {'key': 'value'}
    assert config['abbreviations'] == {**BUILTIN_ABBREVIATIONS, 'custom': 'https://custom.com/{0}'}

    # Test with a non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with an invalid YAML file
    invalid_yaml_file = tmp_path / "invalid.yaml"
    with open(invalid_yaml_file, 'w', encoding='utf-8') as f:
        f.write("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with a YAML file that is not a dict
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    with open(non_dict_yaml_file, 'w', encoding='utf-8') as f:
        f.write("- list item 1\n- list item 2")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #56
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "test_config.yaml"
    config_content = {
        'cookiecutters_dir': '~/test_cookiecutters/',
        'replay_dir': '~/test_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://custom.com/{0}.git'}
    }
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config_content, f)

    result = get_config(config_file)
    assert result['cookiecutters_dir'] == os.path.expanduser('~/test_cookiecutters/')
    assert result['replay_dir'] == os.path.expanduser('~/test_replay/')
    assert result['default_context'] == {'key': 'value'}
    assert result['abbreviations'] == {**BUILTIN_ABBREVIATIONS, 'custom': 'https://custom.com/{0}.git'}

    # Test with a non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with an invalid YAML file
    invalid_yaml_file = tmp_path / "invalid.yaml"
    with open(invalid_yaml_file, 'w', encoding='utf-8') as f:
        f.write("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with a YAML file that is not a dict
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    with open(non_dict_yaml_file, 'w', encoding='utf-8') as f:
        f.write("- list item")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #57
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text("""
cookiecutters_dir: /custom/cookiecutters/
replay_dir: /custom/replay/
default_context:
    key: value
abbreviations:
    custom: https://custom.com/{0}
""")
    config = get_config(config_file)
    assert config['cookiecutters_dir'] == '/custom/cookiecutters/'
    assert config['replay_dir'] == '/custom/replay/'
    assert config['default_context'] == {'key': 'value'}
    assert config['abbreviations']['custom'] == 'https://custom.com/{0}'
    assert config['abbreviations']['gh'] == 'https://github.com/{0}.git'

    # Test with environment variables in paths
    config_file.write_text("""
cookiecutters_dir: $HOME/test_cookiecutters/
replay_dir: ~/test_replay/
""")
    config = get_config(config_file)
    assert config['cookiecutters_dir'] == os.path.expandvars('$HOME/test_cookiecutters/')
    assert config['replay_dir'] == os.path.expanduser('~/test_replay/')

    # Test with non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with invalid YAML
    invalid_yaml_file = tmp_path / "invalid.yaml"
    invalid_yaml_file.write_text("invalid: yaml: content: [unclosed")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with non-dict YAML content
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    non_dict_yaml_file.write_text("- list item")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #58
#--------------------------

```python
def test_get_user_config():
    # Test default config
    config = get_user_config(default_config=True)
    assert config == DEFAULT_CONFIG

    # Test custom default config
    custom_default = {'cookiecutters_dir': '/custom/dir'}
    config = get_user_config(default_config=custom_default)
    expected = merge_configs(DEFAULT_CONFIG, custom_default)
    assert config == expected

    # Test loading from custom config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("cookiecutters_dir: /test/dir\n")
        f.flush()
        config = get_user_config(config_file=f.name)
        assert config['cookiecutters_dir'] == '/test/dir'
        os.unlink(f.name)

    # Test loading from environment variable
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("replay_dir: /env/dir\n")
        f.flush()
        os.environ['COOKIECUTTER_CONFIG'] = f.name
        config = get_user_config()
        assert config['replay_dir'] == '/env/dir'
        del os.environ['COOKIECUTTER_CONFIG']
        os.unlink(f.name)

    # Test loading from default user config path
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("default_context:\n  key: value\n")
        f.flush()
        os.rename(f.name, USER_CONFIG_PATH)
        config = get_user_config()
        assert config['default_context'] == {'key': 'value'}
        os.unlink(USER_CONFIG_PATH)


# LLM-generated content at query #59
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "test_config.yaml"
    config_content = {
        "cookiecutters_dir": "~/test_cookiecutters/",
        "replay_dir": "~/test_replay/",
        "default_context": {"key": "value"},
        "abbreviations": {"custom": "https://custom.com/{0}"}
    }
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump(config_content, f)

    result = get_config(config_file)
    assert result["cookiecutters_dir"] == os.path.expanduser("~/test_cookiecutters/")
    assert result["replay_dir"] == os.path.expanduser("~/test_replay/")
    assert result["default_context"] == {"key": "value"}
    assert result["abbreviations"] == {**BUILTIN_ABBREVIATIONS, "custom": "https://custom.com/{0}"}

    # Test with a non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with an invalid YAML file
    invalid_yaml_file = tmp_path / "invalid.yaml"
    with open(invalid_yaml_file, "w", encoding="utf-8") as f:
        f.write("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with a YAML file that is not a dict
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    with open(non_dict_yaml_file, "w", encoding="utf-8") as f:
        f.write("- not a dict")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #60
#--------------------------

```python
def test_get_user_config():
    # Test default config
    config = get_user_config(default_config=True)
    assert config == DEFAULT_CONFIG

    # Test custom default config
    custom_default = {'cookiecutters_dir': '/custom/dir'}
    config = get_user_config(default_config=custom_default)
    expected = merge_configs(DEFAULT_CONFIG, custom_default)
    assert config == expected

    # Test loading from custom config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("cookiecutters_dir: /test/dir\nabbreviations:\n  custom: 'test'")
        f.flush()
        config = get_user_config(config_file=f.name)
        assert config['cookiecutters_dir'] == '/test/dir'
        assert config['abbreviations']['custom'] == 'test'
        os.unlink(f.name)

    # Test loading from environment variable
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("replay_dir: /env/replay\ncookiecutters_dir: /env/cookies")
        f.flush()
        os.environ['COOKIECUTTER_CONFIG'] = f.name
        config = get_user_config()
        assert config['replay_dir'] == '/env/replay'
        assert config['cookiecutters_dir'] == '/env/cookies'
        del os.environ['COOKIECUTTER_CONFIG']
        os.unlink(f.name)

    # Test loading from default user config path
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("default_context:\n  key: 'value'")
        f.flush()
        os.rename(f.name, USER_CONFIG_PATH)
        config = get_user_config()
        assert config['default_context']['key'] == 'value'
        os.unlink(USER_CONFIG_PATH)


# LLM-generated content at query #61
#--------------------------

```python
def test_get_user_config():
    # Test default config
    config = get_user_config(default_config=True)
    assert config == DEFAULT_CONFIG

    # Test custom default config
    custom_default = {'cookiecutters_dir': '/custom/dir'}
    config = get_user_config(default_config=custom_default)
    expected = merge_configs(DEFAULT_CONFIG, custom_default)
    assert config == expected

    # Test loading from custom config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump({'cookiecutters_dir': '/test/dir'}, f)
        f.flush()
        config = get_user_config(config_file=f.name)
        assert config['cookiecutters_dir'] == '/test/dir'
        os.unlink(f.name)

    # Test COOKIECUTTER_CONFIG environment variable
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump({'replay_dir': '/env/dir'}, f)
        f.flush()
        os.environ['COOKIECUTTER_CONFIG'] = f.name
        config = get_user_config()
        assert config['replay_dir'] == '/env/dir'
        del os.environ['COOKIECUTTER_CONFIG']
        os.unlink(f.name)

    # Test loading from USER_CONFIG_PATH if it exists
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump({'default_context': {'key': 'value'}}, f)
        f.flush()
        os.rename(f.name, USER_CONFIG_PATH)
        config = get_user_config()
        assert config['default_context'] == {'key': 'value'}
        os.unlink(USER_CONFIG_PATH)


# LLM-generated content at query #62
#--------------------------

```python
def test_get_user_config():
    # Test default config
    config = get_user_config(default_config=True)
    assert config == DEFAULT_CONFIG

    # Test custom default config
    custom_default = {'cookiecutters_dir': '/custom/dir'}
    config = get_user_config(default_config=custom_default)
    expected = merge_configs(DEFAULT_CONFIG, custom_default)
    assert config == expected

    # Test loading from custom config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("cookiecutters_dir: /test/dir\nabbreviations:\n  custom: 'test'")
        f.flush()
        config = get_user_config(config_file=f.name)
        assert config['cookiecutters_dir'] == '/test/dir'
        assert config['abbreviations']['custom'] == 'test'
        os.unlink(f.name)

    # Test COOKIECUTTER_CONFIG environment variable
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("replay_dir: /env/dir")
        f.flush()
        os.environ['COOKIECUTTER_CONFIG'] = f.name
        config = get_user_config()
        assert config['replay_dir'] == '/env/dir'
        del os.environ['COOKIECUTTER_CONFIG']
        os.unlink(f.name)

    # Test loading from default user config path
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("default_context:\n  key: 'value'")
        f.flush()
        os.replace(f.name, USER_CONFIG_PATH)
        config = get_user_config()
        assert config['default_context']['key'] == 'value'
        os.unlink(USER_CONFIG_PATH)


# LLM-generated content at query #63
#--------------------------

```python
def test_get_user_config():
    # Test 1: default_config is True
    config = get_user_config(default_config=True)
    assert config == DEFAULT_CONFIG

    # Test 2: default_config is a dict
    custom_config = {'cookiecutters_dir': '/custom/dir'}
    config = get_user_config(default_config=custom_config)
    expected = merge_configs(DEFAULT_CONFIG, custom_config)
    assert config == expected

    # Test 3: config_file is specified and exists
    with patch('builtins.open', mock_open(read_data='cookiecutters_dir: /test/dir')):
        with patch('os.path.exists', return_value=True):
            config = get_user_config(config_file='/test/config')
            assert config['cookiecutters_dir'] == '/test/dir'

    # Test 4: config_file is specified but does not exist
    with patch('os.path.exists', return_value=False):
        with pytest.raises(ConfigDoesNotExistException):
            get_user_config(config_file='/nonexistent/config')

    # Test 5: COOKIECUTTER_CONFIG environment variable is set
    with patch.dict('os.environ', {'COOKIECUTTER_CONFIG': '/env/config'}):
        with patch('builtins.open', mock_open(read_data='cookiecutters_dir: /env/dir')):
            with patch('os.path.exists', return_value=True):
                config = get_user_config()
                assert config['cookiecutters_dir'] == '/env/dir'

    # Test 6: COOKIECUTTER_CONFIG environment variable is set but file does not exist
    with patch.dict('os.environ', {'COOKIECUTTER_CONFIG': '/nonexistent/config'}):
        with patch('os.path.exists', return_value=False):
            with pytest.raises(ConfigDoesNotExistException):
                get_user_config()

    # Test 7: USER_CONFIG_PATH exists
    with patch('os.path.exists', return_value=True):
        with patch('builtins.open', mock_open(read_data='cookiecutters_dir: /user/dir')):
            config = get_user_config()
            assert config['cookiecutters_dir'] == '/user/dir'

    # Test 8: USER_CONFIG_PATH does not exist
    with patch('os.path.exists', return_value=False):
        config = get_user_config()
        assert config == DEFAULT_CONFIG


# LLM-generated content at query #64
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / 'config.yaml'
    config_content = {
        'cookiecutters_dir': '~/test_cookiecutters/',
        'replay_dir': '~/test_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://custom.com/{0}'}
    }
    with open(config_file, 'w') as f:
        yaml.dump(config_content, f)

    result = get_config(config_file)
    assert result['cookiecutters_dir'] == os.path.expanduser('~/test_cookiecutters/')
    assert result['replay_dir'] == os.path.expanduser('~/test_replay/')
    assert result['default_context'] == {'key': 'value'}
    assert result['abbreviations'] == {**BUILTIN_ABBREVIATIONS, 'custom': 'https://custom.com/{0}'}

    # Test with non-existent config file
    non_existent_file = tmp_path / 'non_existent.yaml'
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with invalid YAML
    invalid_yaml_file = tmp_path / 'invalid.yaml'
    with open(invalid_yaml_file, 'w') as f:
        f.write('invalid: yaml: content: [')
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with non-dict YAML
    non_dict_yaml_file = tmp_path / 'non_dict.yaml'
    with open(non_dict_yaml_file, 'w') as f:
        f.write('- list item')
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #65
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "test_config.yaml"
    config_content = {
        'cookiecutters_dir': '~/test_cookiecutters/',
        'replay_dir': '~/test_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'test': 'test_url'}
    }
    with open(config_file, 'w') as f:
        yaml.dump(config_content, f)

    result = get_config(config_file)
    assert result['cookiecutters_dir'] == os.path.expanduser('~/test_cookiecutters/')
    assert result['replay_dir'] == os.path.expanduser('~/test_replay/')
    assert result['default_context'] == {'key': 'value'}
    assert result['abbreviations'] == {**BUILTIN_ABBREVIATIONS, 'test': 'test_url'}

    # Test with a non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with an invalid YAML file
    invalid_yaml_file = tmp_path / "invalid.yaml"
    with open(invalid_yaml_file, 'w') as f:
        f.write("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with a YAML file that is not a dict
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    with open(non_dict_yaml_file, 'w') as f:
        f.write("- list item")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #66
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "test_config.yaml"
    config_data = {
        'cookiecutters_dir': '~/test_cookiecutters/',
        'replay_dir': '~/test_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://custom.com/{0}'}
    }
    with open(config_file, 'w') as f:
        yaml.dump(config_data, f)

    result = get_config(config_file)
    assert result['cookiecutters_dir'] == os.path.expanduser('~/test_cookiecutters/')
    assert result['replay_dir'] == os.path.expanduser('~/test_replay/')
    assert result['default_context'] == {'key': 'value'}
    assert result['abbreviations'] == {**BUILTIN_ABBREVIATIONS, 'custom': 'https://custom.com/{0}'}

    # Test with a non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with an invalid YAML file
    invalid_yaml_file = tmp_path / "invalid.yaml"
    with open(invalid_yaml_file, 'w') as f:
        f.write("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with a YAML file that is not a dict
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    with open(non_dict_yaml_file, 'w') as f:
        f.write("- list item 1\n- list item 2")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #67
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "test_config.yaml"
    config_content = {
        'cookiecutters_dir': '~/test_cookiecutters/',
        'replay_dir': '~/test_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://custom.com/{0}'}
    }
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config_content, f)

    config = get_config(config_file)
    assert config['cookiecutters_dir'] == os.path.expanduser('~/test_cookiecutters/')
    assert config['replay_dir'] == os.path.expanduser('~/test_replay/')
    assert config['default_context'] == {'key': 'value'}
    assert config['abbreviations'] == {**BUILTIN_ABBREVIATIONS, 'custom': 'https://custom.com/{0}'}

    # Test with a non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with an invalid YAML file
    invalid_yaml_file = tmp_path / "invalid.yaml"
    with open(invalid_yaml_file, 'w', encoding='utf-8') as f:
        f.write("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with a YAML file that is not a dict
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    with open(non_dict_yaml_file, 'w', encoding='utf-8') as f:
        f.write("- list item")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #68
#--------------------------

```python
def test_get_user_config():
    # Test default config
    config = get_user_config(default_config=True)
    assert config == DEFAULT_CONFIG

    # Test custom config dict
    custom_config = {'cookiecutters_dir': '/custom/dir'}
    merged_config = merge_configs(DEFAULT_CONFIG, custom_config)
    config = get_user_config(default_config=custom_config)
    assert config == merged_config

    # Test loading from custom config file
    with patch('builtins.open', mock_open(read_data='cookiecutters_dir: /test/dir')):
        with patch('os.path.exists', return_value=True):
            config = get_user_config(config_file='/test/config')
            assert config['cookiecutters_dir'] == '/test/dir'

    # Test loading from environment variable
    with patch.dict('os.environ', {'COOKIECUTTER_CONFIG': '/env/config'}):
        with patch('builtins.open', mock_open(read_data='cookiecutters_dir: /env/dir')):
            config = get_user_config()
            assert config['cookiecutters_dir'] == '/env/dir'

    # Test loading from default user config path
    with patch('os.path.exists', return_value=True):
        with patch('builtins.open', mock_open(read_data='cookiecutters_dir: /user/dir')):
            config = get_user_config()
            assert config['cookiecutters_dir'] == '/user/dir'

    # Test fallback to default config when no config file exists
    with patch('os.path.exists', return_value=False):
        config = get_user_config()
        assert config == DEFAULT_CONFIG


# LLM-generated content at query #69
#--------------------------

```python
def test_get_user_config():
    # Test default config
    config = get_user_config(default_config=True)
    assert config == DEFAULT_CONFIG

    # Test custom config dict
    custom_config = {'cookiecutters_dir': '/custom/dir'}
    config = get_user_config(default_config=custom_config)
    expected = merge_configs(DEFAULT_CONFIG, custom_config)
    assert config == expected

    # Test loading from custom config file
    with patch('builtins.open', mock_open(read_data='cookiecutters_dir: /test/dir')):
        with patch('os.path.exists', return_value=True):
            config = get_user_config(config_file='/test/config')
            assert config['cookiecutters_dir'] == '/test/dir'

    # Test loading from environment variable
    with patch.dict('os.environ', {'COOKIECUTTER_CONFIG': '/env/config'}):
        with patch('builtins.open', mock_open(read_data='cookiecutters_dir: /env/dir')):
            with patch('os.path.exists', return_value=True):
                config = get_user_config()
                assert config['cookiecutters_dir'] == '/env/dir'

    # Test loading from default user config path
    with patch('os.path.exists', return_value=True):
        with patch('builtins.open', mock_open(read_data='cookiecutters_dir: /user/dir')):
            config = get_user_config()
            assert config['cookiecutters_dir'] == '/user/dir'

    # Test fallback to default config when no config file exists
    with patch('os.path.exists', return_value=False):
        config = get_user_config()
        assert config == DEFAULT_CONFIG


# LLM-generated content at query #70
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "test_config.yaml"
    config_content = {
        'cookiecutters_dir': '~/test_cookiecutters/',
        'replay_dir': '~/test_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://custom.com/{0}'}
    }
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config_content, f)

    result = get_config(config_file)
    assert result['cookiecutters_dir'] == os.path.expanduser('~/test_cookiecutters/')
    assert result['replay_dir'] == os.path.expanduser('~/test_replay/')
    assert result['default_context'] == {'key': 'value'}
    assert result['abbreviations'] == {**BUILTIN_ABBREVIATIONS, 'custom': 'https://custom.com/{0}'}

    # Test with a non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with an invalid YAML file
    invalid_yaml_file = tmp_path / "invalid.yaml"
    with open(invalid_yaml_file, 'w', encoding='utf-8') as f:
        f.write("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with a YAML file that is not a dict
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    with open(non_dict_yaml_file, 'w', encoding='utf-8') as f:
        f.write("- list item 1\n- list item 2")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #71
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "test_config.yaml"
    config_content = {
        'cookiecutters_dir': '~/test_cookiecutters/',
        'replay_dir': '~/test_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'test': 'https://test.com/{0}'}
    }
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config_content, f)

    result = get_config(config_file)
    assert result['cookiecutters_dir'] == os.path.expanduser('~/test_cookiecutters/')
    assert result['replay_dir'] == os.path.expanduser('~/test_replay/')
    assert result['default_context'] == {'key': 'value'}
    assert result['abbreviations'] == {**BUILTIN_ABBREVIATIONS, 'test': 'https://test.com/{0}'}

    # Test with a non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with an invalid YAML file
    invalid_yaml_file = tmp_path / "invalid.yaml"
    with open(invalid_yaml_file, 'w', encoding='utf-8') as f:
        f.write("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with a YAML file that is not a dict
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    with open(non_dict_yaml_file, 'w', encoding='utf-8') as f:
        f.write("- list item 1\n- list item 2")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #72
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / 'config.yaml'
    config_file.write_text('''
cookiecutters_dir: /custom/cookiecutters/
replay_dir: /custom/replay/
default_context:
    key: value
abbreviations:
    custom: https://custom.com/{0}
''')

    config = get_config(config_file)
    assert config['cookiecutters_dir'] == '/custom/cookiecutters/'
    assert config['replay_dir'] == '/custom/replay/'
    assert config['default_context'] == {'key': 'value'}
    assert config['abbreviations']['custom'] == 'https://custom.com/{0}'
    assert config['abbreviations']['gh'] == 'https://github.com/{0}.git'

    # Test with environment variable expansion
    config_file.write_text('''
cookiecutters_dir: $HOME/test/
replay_dir: ~/test/
''')
    config = get_config(config_file)
    assert config['cookiecutters_dir'] == os.path.expandvars('$HOME/test/')
    assert config['replay_dir'] == os.path.expanduser('~/test/')

    # Test with non-existent config file
    non_existent_file = tmp_path / 'non_existent.yaml'
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with invalid YAML
    invalid_yaml_file = tmp_path / 'invalid.yaml'
    invalid_yaml_file.write_text('invalid: yaml: content: [unclosed')
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with non-dict YAML content
    non_dict_yaml_file = tmp_path / 'non_dict.yaml'
    non_dict_yaml_file.write_text('- list item')
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #73
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "test_config.yaml"
    config_content = {
        'cookiecutters_dir': '~/test_cookiecutters/',
        'replay_dir': '~/test_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://custom.com/{0}'}
    }
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config_content, f)

    result = get_config(config_file)
    assert result['cookiecutters_dir'] == os.path.expanduser('~/test_cookiecutters/')
    assert result['replay_dir'] == os.path.expanduser('~/test_replay/')
    assert result['default_context'] == {'key': 'value'}
    assert result['abbreviations'] == {**BUILTIN_ABBREVIATIONS, 'custom': 'https://custom.com/{0}'}

    # Test with a non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with an invalid YAML file
    invalid_yaml_file = tmp_path / "invalid.yaml"
    with open(invalid_yaml_file, 'w', encoding='utf-8') as f:
        f.write("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with a YAML file that is not a dict
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    with open(non_dict_yaml_file, 'w', encoding='utf-8') as f:
        f.write("- list item")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #74
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "test_config.yaml"
    config_content = {
        "cookiecutters_dir": "~/test_cookiecutters/",
        "replay_dir": "~/test_replay/",
        "default_context": {"key": "value"},
        "abbreviations": {"custom": "https://custom.com/{0}"}
    }
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump(config_content, f)

    result = get_config(config_file)

    assert result["cookiecutters_dir"] == os.path.expanduser("~/test_cookiecutters/")
    assert result["replay_dir"] == os.path.expanduser("~/test_replay/")
    assert result["default_context"] == {"key": "value"}
    assert result["abbreviations"] == {
        "gh": "https://github.com/{0}.git",
        "gl": "https://gitlab.com/{0}.git",
        "bb": "https://bitbucket.org/{0}",
        "custom": "https://custom.com/{0}"
    }

    # Test with a non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with an invalid YAML file
    invalid_yaml_file = tmp_path / "invalid.yaml"
    with open(invalid_yaml_file, "w", encoding="utf-8") as f:
        f.write("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with a YAML file that is not a dict
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    with open(non_dict_yaml_file, "w", encoding="utf-8") as f:
        f.write("- list item 1\n- list item 2")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #75
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "test_config.yaml"
    config_content = {
        'cookiecutters_dir': '~/test_cookiecutters/',
        'replay_dir': '~/test_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://custom.com/{0}'}
    }
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config_content, f)

    config = get_config(config_file)
    assert config['cookiecutters_dir'] == os.path.expanduser('~/test_cookiecutters/')
    assert config['replay_dir'] == os.path.expanduser('~/test_replay/')
    assert config['default_context'] == {'key': 'value'}
    assert config['abbreviations'] == {**BUILTIN_ABBREVIATIONS, 'custom': 'https://custom.com/{0}'}

    # Test with a non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with an invalid YAML file
    invalid_yaml_file = tmp_path / "invalid.yaml"
    with open(invalid_yaml_file, 'w', encoding='utf-8') as f:
        f.write("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with a YAML file that is not a dict
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    with open(non_dict_yaml_file, 'w', encoding='utf-8') as f:
        f.write("- list item")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "test_config.yaml"
    config_content = {
        'cookiecutters_dir': '~/test_cookiecutters/',
        'replay_dir': '~/test_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://custom.com/{0}'}
    }
    with open(config_file, 'w') as f:
        yaml.dump(config_content, f)

    result = get_config(config_file)
    assert result['cookiecutters_dir'] == os.path.expanduser('~/test_cookiecutters/')
    assert result['replay_dir'] == os.path.expanduser('~/test_replay/')
    assert result['default_context'] == {'key': 'value'}
    assert result['abbreviations'] == {**BUILTIN_ABBREVIATIONS, 'custom': 'https://custom.com/{0}'}

    # Test with a non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with an invalid YAML file
    invalid_yaml_file = tmp_path / "invalid.yaml"
    with open(invalid_yaml_file, 'w') as f:
        f.write("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with a YAML file that is not a dict
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    with open(non_dict_yaml_file, 'w') as f:
        f.write("- list_item1\n- list_item2")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #2
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / 'config.yaml'
    config_file.write_text('''
cookiecutters_dir: /custom/cookiecutters/
replay_dir: /custom/replay/
default_context:
    key1: value1
abbreviations:
    custom: https://custom.com/{0}
''')

    config = get_config(config_file)
    assert config['cookiecutters_dir'] == '/custom/cookiecutters/'
    assert config['replay_dir'] == '/custom/replay/'
    assert config['default_context'] == {'key1': 'value1'}
    assert config['abbreviations']['custom'] == 'https://custom.com/{0}'
    assert config['abbreviations']['gh'] == 'https://github.com/{0}.git'

    # Test with environment variable expansion
    config_file.write_text('''
cookiecutters_dir: $HOME/test/
replay_dir: ~/test/
''')
    config = get_config(config_file)
    assert config['cookiecutters_dir'] == os.path.expandvars('$HOME/test/')
    assert config['replay_dir'] == os.path.expanduser('~/test/')

    # Test with non-existent config file
    non_existent_file = tmp_path / 'non_existent.yaml'
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with invalid YAML
    invalid_yaml_file = tmp_path / 'invalid.yaml'
    invalid_yaml_file.write_text('invalid: yaml: content: [')
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with non-dict YAML content
    non_dict_yaml_file = tmp_path / 'non_dict.yaml'
    non_dict_yaml_file.write_text('- list item')
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #3
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "test_config.yaml"
    config_content = {
        'cookiecutters_dir': '~/test_cookiecutters/',
        'replay_dir': '~/test_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://custom.com/{0}'}
    }
    with open(config_file, 'w') as f:
        yaml.dump(config_content, f)

    result = get_config(config_file)

    assert result['cookiecutters_dir'] == os.path.expanduser('~/test_cookiecutters/')
    assert result['replay_dir'] == os.path.expanduser('~/test_replay/')
    assert result['default_context'] == {'key': 'value'}
    assert result['abbreviations'] == {**BUILTIN_ABBREVIATIONS, 'custom': 'https://custom.com/{0}'}

    # Test with a non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with an invalid YAML file
    invalid_yaml_file = tmp_path / "invalid.yaml"
    with open(invalid_yaml_file, 'w') as f:
        f.write("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with a YAML file that is not a dict
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    with open(non_dict_yaml_file, 'w') as f:
        f.write("- list item 1\n- list item 2")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #4
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "config.yaml"
    config_content = {
        'cookiecutters_dir': '~/test_cookiecutters/',
        'replay_dir': '~/test_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://custom.com/{0}'}
    }
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config_content, f)

    result = get_config(config_file)
    assert result['cookiecutters_dir'] == os.path.expanduser('~/test_cookiecutters/')
    assert result['replay_dir'] == os.path.expanduser('~/test_replay/')
    assert result['default_context'] == {'key': 'value'}
    assert result['abbreviations'] == {**BUILTIN_ABBREVIATIONS, **{'custom': 'https://custom.com/{0}'}}

    # Test with a non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with an invalid YAML file
    invalid_yaml_file = tmp_path / "invalid.yaml"
    with open(invalid_yaml_file, 'w', encoding='utf-8') as f:
        f.write("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with a YAML file that is not a dict
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    with open(non_dict_yaml_file, 'w', encoding='utf-8') as f:
        f.write("- list item")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #5
#--------------------------

```python
def test_get_user_config():
    # Test default config
    config = get_user_config(default_config=True)
    assert config == DEFAULT_CONFIG

    # Test custom default config
    custom_default = {'cookiecutters_dir': '/custom/dir'}
    config = get_user_config(default_config=custom_default)
    expected = merge_configs(DEFAULT_CONFIG, custom_default)
    assert config == expected

    # Test loading from custom config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("cookiecutters_dir: /test/dir\nabbreviations:\n  custom: 'test'")
        f.flush()
        config = get_user_config(config_file=f.name)
        assert config['cookiecutters_dir'] == '/test/dir'
        assert config['abbreviations']['custom'] == 'test'
        os.unlink(f.name)

    # Test COOKIECUTTER_CONFIG environment variable
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("replay_dir: /env/dir")
        f.flush()
        os.environ['COOKIECUTTER_CONFIG'] = f.name
        config = get_user_config()
        assert config['replay_dir'] == '/env/dir'
        del os.environ['COOKIECUTTER_CONFIG']
        os.unlink(f.name)

    # Test loading from default user config path
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("default_context:\n  key: 'value'")
        f.flush()
        os.rename(f.name, USER_CONFIG_PATH)
        config = get_user_config()
        assert config['default_context']['key'] == 'value'
        os.unlink(USER_CONFIG_PATH)


# LLM-generated content at query #6
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "config.yaml"
    config_file.write_text("""
cookiecutters_dir: /custom/cookiecutters/
replay_dir: /custom/replay/
default_context:
    key: value
abbreviations:
    custom: https://custom.com/{0}
""")

    config = get_config(config_file)
    assert config['cookiecutters_dir'] == '/custom/cookiecutters/'
    assert config['replay_dir'] == '/custom/replay/'
    assert config['default_context'] == {'key': 'value'}
    assert config['abbreviations']['custom'] == 'https://custom.com/{0}'
    assert config['abbreviations']['gh'] == 'https://github.com/{0}.git'

    # Test with environment variable expansion
    config_file.write_text("""
cookiecutters_dir: $HOME/test/
replay_dir: ~/test/
""")
    config = get_config(config_file)
    assert config['cookiecutters_dir'] == os.path.expanduser('~') + '/test/'
    assert config['replay_dir'] == os.path.expanduser('~') + '/test/'

    # Test with non-existent config file
    with pytest.raises(ConfigDoesNotExistException):
        get_config(tmp_path / "nonexistent.yaml")

    # Test with invalid YAML
    config_file.write_text("invalid yaml: [")
    with pytest.raises(InvalidConfiguration):
        get_config(config_file)

    # Test with non-dict YAML
    config_file.write_text("not a dict")
    with pytest.raises(InvalidConfiguration):
        get_config(config_file)


# LLM-generated content at query #7
#--------------------------

```python
def test_get_user_config():
    # Test default config
    config = get_user_config(default_config=True)
    assert config == DEFAULT_CONFIG

    # Test custom config dict
    custom_config = {'cookiecutters_dir': '/custom/dir'}
    config = get_user_config(default_config=custom_config)
    expected = merge_configs(DEFAULT_CONFIG, custom_config)
    assert config == expected

    # Test loading from custom config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump({'cookiecutters_dir': '/test/dir'}, f)
        f.flush()
        config = get_user_config(config_file=f.name)
        assert config['cookiecutters_dir'] == '/test/dir'
        os.unlink(f.name)

    # Test loading from environment variable
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump({'replay_dir': '/env/dir'}, f)
        f.flush()
        os.environ['COOKIECUTTER_CONFIG'] = f.name
        config = get_user_config()
        assert config['replay_dir'] == '/env/dir'
        del os.environ['COOKIECUTTER_CONFIG']
        os.unlink(f.name)

    # Test loading from default user config path
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump({'default_context': {'key': 'value'}}, f)
        f.flush()
        with patch('cookiecutter.config.USER_CONFIG_PATH', f.name):
            config = get_user_config()
            assert config['default_context'] == {'key': 'value'}
        os.unlink(f.name)

    # Test non-existent config file raises exception
    with pytest.raises(ConfigDoesNotExistException):
        get_user_config(config_file='/nonexistent/path.yaml')

    # Test invalid YAML raises exception
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("invalid: yaml: content")
        f.flush()
        with pytest.raises(InvalidConfiguration):
            get_user_config(config_file=f.name)
        os.unlink(f.name)


# LLM-generated content at query #8
#--------------------------

```python
def test_get_user_config():
    # Test default config
    config = get_user_config(default_config=True)
    assert config == DEFAULT_CONFIG

    # Test custom config dict
    custom_config = {'cookiecutters_dir': '/custom/dir'}
    config = get_user_config(default_config=custom_config)
    expected = merge_configs(DEFAULT_CONFIG, custom_config)
    assert config == expected

    # Test custom config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump({'cookiecutters_dir': '/custom/dir'}, f)
        config = get_user_config(config_file=f.name)
        expected = merge_configs(DEFAULT_CONFIG, {'cookiecutters_dir': '/custom/dir'})
        assert config == expected

    # Test environment variable
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump({'cookiecutters_dir': '/env/dir'}, f)
        os.environ['COOKIECUTTER_CONFIG'] = f.name
        config = get_user_config()
        expected = merge_configs(DEFAULT_CONFIG, {'cookiecutters_dir': '/env/dir'})
        assert config == expected
        del os.environ['COOKIECUTTER_CONFIG']

    # Test default config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump({'cookiecutters_dir': '/default/dir'}, f)
        os.rename(f.name, USER_CONFIG_PATH)
        config = get_user_config()
        expected = merge_configs(DEFAULT_CONFIG, {'cookiecutters_dir': '/default/dir'})
        assert config == expected
        os.remove(USER_CONFIG_PATH)


# LLM-generated content at query #9
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "test_config.yaml"
    config_content = {
        'cookiecutters_dir': '~/test_cookiecutters/',
        'replay_dir': '~/test_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://custom.com/{0}'}
    }
    with open(config_file, 'w') as f:
        yaml.dump(config_content, f)

    result = get_config(config_file)
    assert result['cookiecutters_dir'] == os.path.expanduser('~/test_cookiecutters/')
    assert result['replay_dir'] == os.path.expanduser('~/test_replay/')
    assert result['default_context'] == {'key': 'value'}
    assert result['abbreviations'] == {**BUILTIN_ABBREVIATIONS, 'custom': 'https://custom.com/{0}'}

    # Test with a non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with an invalid YAML file
    invalid_yaml_file = tmp_path / "invalid.yaml"
    with open(invalid_yaml_file, 'w') as f:
        f.write("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with a YAML file that is not a dict
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    with open(non_dict_yaml_file, 'w') as f:
        f.write("- list item")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #10
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "test_config.yaml"
    config_content = {
        'cookiecutters_dir': '~/test_cookiecutters/',
        'replay_dir': '~/test_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://custom.com/{0}'}
    }
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config_content, f)

    result = get_config(config_file)
    assert result['cookiecutters_dir'] == os.path.expanduser('~/test_cookiecutters/')
    assert result['replay_dir'] == os.path.expanduser('~/test_replay/')
    assert result['default_context'] == {'key': 'value'}
    assert result['abbreviations'] == {**BUILTIN_ABBREVIATIONS, 'custom': 'https://custom.com/{0}'}

    # Test with a non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with an invalid YAML file
    invalid_yaml_file = tmp_path / "invalid.yaml"
    with open(invalid_yaml_file, 'w', encoding='utf-8') as f:
        f.write("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with a YAML file that is not a dict
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    with open(non_dict_yaml_file, 'w', encoding='utf-8') as f:
        f.write("- list item")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #11
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "test_config.yaml"
    config_content = {
        "cookiecutters_dir": "~/test_cookiecutters/",
        "replay_dir": "~/test_replay/",
        "default_context": {"key": "value"},
        "abbreviations": {"custom": "https://custom.com/{0}"}
    }
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump(config_content, f)

    result = get_config(config_file)
    assert result["cookiecutters_dir"] == os.path.expanduser("~/test_cookiecutters/")
    assert result["replay_dir"] == os.path.expanduser("~/test_replay/")
    assert result["default_context"] == {"key": "value"}
    assert result["abbreviations"] == {
        "gh": "https://github.com/{0}.git",
        "gl": "https://gitlab.com/{0}.git",
        "bb": "https://bitbucket.org/{0}",
        "custom": "https://custom.com/{0}"
    }

    # Test with a non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with an invalid YAML file
    invalid_yaml_file = tmp_path / "invalid.yaml"
    with open(invalid_yaml_file, "w", encoding="utf-8") as f:
        f.write("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with a YAML file that is not a dict
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    with open(non_dict_yaml_file, "w", encoding="utf-8") as f:
        f.write("- list item")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #12
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "test_config.yaml"
    config_content = {
        'cookiecutters_dir': '~/test_cookiecutters/',
        'replay_dir': '~/test_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://custom.com/{0}'}
    }
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config_content, f)

    config = get_config(config_file)
    assert config['cookiecutters_dir'] == os.path.expanduser('~/test_cookiecutters/')
    assert config['replay_dir'] == os.path.expanduser('~/test_replay/')
    assert config['default_context'] == {'key': 'value'}
    assert config['abbreviations'] == {**BUILTIN_ABBREVIATIONS, 'custom': 'https://custom.com/{0}'}

    # Test with a non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with an invalid YAML file
    invalid_yaml_file = tmp_path / "invalid.yaml"
    with open(invalid_yaml_file, 'w', encoding='utf-8') as f:
        f.write("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with a YAML file that is not a dict
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    with open(non_dict_yaml_file, 'w', encoding='utf-8') as f:
        f.write("- list_item1\n- list_item2")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #13
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "test_config.yaml"
    config_content = {
        "cookiecutters_dir": "~/test_cookiecutters/",
        "replay_dir": "~/test_replay/",
        "default_context": {"key": "value"},
        "abbreviations": {"custom": "https://custom.com/{0}"}
    }
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump(config_content, f)

    result = get_config(config_file)
    assert result["cookiecutters_dir"] == os.path.expanduser("~/test_cookiecutters/")
    assert result["replay_dir"] == os.path.expanduser("~/test_replay/")
    assert result["default_context"] == {"key": "value"}
    assert result["abbreviations"] == {**BUILTIN_ABBREVIATIONS, "custom": "https://custom.com/{0}"}

    # Test with a non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with an invalid YAML file
    invalid_yaml_file = tmp_path / "invalid.yaml"
    with open(invalid_yaml_file, "w", encoding="utf-8") as f:
        f.write("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with a YAML file that is not a dict
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    with open(non_dict_yaml_file, "w", encoding="utf-8") as f:
        f.write("- list_item1\n- list_item2")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #14
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "test_config.yaml"
    config_content = {
        'cookiecutters_dir': '~/test_cookiecutters/',
        'replay_dir': '~/test_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://custom.com/{0}'}
    }
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config_content, f)

    result = get_config(config_file)
    assert result['cookiecutters_dir'] == os.path.expanduser('~/test_cookiecutters/')
    assert result['replay_dir'] == os.path.expanduser('~/test_replay/')
    assert result['default_context'] == {'key': 'value'}
    assert result['abbreviations'] == {**BUILTIN_ABBREVIATIONS, 'custom': 'https://custom.com/{0}'}

    # Test with a non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with an invalid YAML file
    invalid_yaml_file = tmp_path / "invalid.yaml"
    with open(invalid_yaml_file, 'w', encoding='utf-8') as f:
        f.write("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with a YAML file that is not a dict
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    with open(non_dict_yaml_file, 'w', encoding='utf-8') as f:
        f.write("- list item 1\n- list item 2")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #15
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / 'config.yaml'
    config_file.write_text('''
cookiecutters_dir: /custom/cookiecutters/
replay_dir: /custom/replay/
default_context:
    key: value
abbreviations:
    custom: https://custom.com/{0}
''')
    config = get_config(config_file)
    assert config['cookiecutters_dir'] == '/custom/cookiecutters/'
    assert config['replay_dir'] == '/custom/replay/'
    assert config['default_context'] == {'key': 'value'}
    assert config['abbreviations']['custom'] == 'https://custom.com/{0}'
    assert config['abbreviations']['gh'] == 'https://github.com/{0}.git'

    # Test with environment variable expansion
    config_file.write_text('''
cookiecutters_dir: $HOME/test/
replay_dir: ~/replay/
''')
    config = get_config(config_file)
    assert config['cookiecutters_dir'] == os.path.expanduser('~') + '/test/'
    assert config['replay_dir'] == os.path.expanduser('~/replay/')

    # Test with non-existent config file
    non_existent_file = tmp_path / 'non_existent.yaml'
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with invalid YAML
    invalid_yaml_file = tmp_path / 'invalid.yaml'
    invalid_yaml_file.write_text('invalid: yaml: content: [unclosed')
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with non-dict YAML content
    non_dict_yaml_file = tmp_path / 'non_dict.yaml'
    non_dict_yaml_file.write_text('- list item')
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #16
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / 'config.yaml'
    config_content = {
        'cookiecutters_dir': '~/test_cookiecutters/',
        'replay_dir': '~/test_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://custom.com/{0}'}
    }
    config_file.write_text(yaml.dump(config_content))

    result = get_config(config_file)
    assert result['cookiecutters_dir'] == os.path.expanduser('~/test_cookiecutters/')
    assert result['replay_dir'] == os.path.expanduser('~/test_replay/')
    assert result['default_context'] == {'key': 'value'}
    assert result['abbreviations'] == {**BUILTIN_ABBREVIATIONS, 'custom': 'https://custom.com/{0}'}

    # Test with a non-existent config file
    non_existent_file = tmp_path / 'non_existent.yaml'
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with an invalid YAML file
    invalid_yaml_file = tmp_path / 'invalid.yaml'
    invalid_yaml_file.write_text('invalid: yaml: content: [')
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with a YAML file that is not a dict
    non_dict_yaml_file = tmp_path / 'non_dict.yaml'
    non_dict_yaml_file.write_text('- list item')
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #17
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "test_config.yaml"
    config_content = {
        'cookiecutters_dir': '~/test_cookiecutters/',
        'replay_dir': '~/test_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://custom.com/{0}'}
    }
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config_content, f)

    result = get_config(config_file)

    assert result['cookiecutters_dir'] == os.path.expanduser('~/test_cookiecutters/')
    assert result['replay_dir'] == os.path.expanduser('~/test_replay/')
    assert result['default_context'] == {'key': 'value'}
    assert result['abbreviations'] == {**BUILTIN_ABBREVIATIONS, 'custom': 'https://custom.com/{0}'}

    # Test with a non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with an invalid YAML file
    invalid_yaml_file = tmp_path / "invalid.yaml"
    with open(invalid_yaml_file, 'w', encoding='utf-8') as f:
        f.write("invalid: yaml: content: [")

    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with a YAML file that has a non-dict top-level element
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    with open(non_dict_yaml_file, 'w', encoding='utf-8') as f:
        f.write("- list item")

    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #18
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "config.yaml"
    config_content = {
        'cookiecutters_dir': '~/test_cookiecutters/',
        'replay_dir': '~/test_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://custom.com/{0}'}
    }
    config_file.write_text(yaml.dump(config_content))

    result = get_config(config_file)
    assert result['cookiecutters_dir'] == os.path.expanduser('~/test_cookiecutters/')
    assert result['replay_dir'] == os.path.expanduser('~/test_replay/')
    assert result['default_context'] == {'key': 'value'}
    assert result['abbreviations'] == {**BUILTIN_ABBREVIATIONS, 'custom': 'https://custom.com/{0}'}

    # Test with a non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with an invalid YAML file
    invalid_yaml_file = tmp_path / "invalid.yaml"
    invalid_yaml_file.write_text("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with a YAML file that is not a dict
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    non_dict_yaml_file.write_text("- list item")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #19
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "cookiecutterrc"
    config_content = {
        'cookiecutters_dir': '~/custom_cookiecutters/',
        'replay_dir': '~/custom_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://custom.com/{0}'}
    }
    config_file.write_text(yaml.dump(config_content))

    result = get_config(config_file)

    assert result['cookiecutters_dir'] == os.path.expanduser('~/custom_cookiecutters/')
    assert result['replay_dir'] == os.path.expanduser('~/custom_replay/')
    assert result['default_context'] == {'key': 'value'}
    assert result['abbreviations'] == {
        'gh': 'https://github.com/{0}.git',
        'gl': 'https://gitlab.com/{0}.git',
        'bb': 'https://bitbucket.org/{0}',
        'custom': 'https://custom.com/{0}'
    }

def test_get_config_nonexistent_file():
    # Test with a non-existent config file
    with pytest.raises(ConfigDoesNotExistException):
        get_config('nonexistent_file.yaml')

def test_get_config_invalid_yaml(tmp_path):
    # Test with an invalid YAML file
    config_file = tmp_path / "invalid_config.yaml"
    config_file.write_text("invalid: yaml: content: [unclosed")

    with pytest.raises(InvalidConfiguration):
        get_config(config_file)

def test_get_config_non_dict_yaml(tmp_path):
    # Test with a YAML file that doesn't contain a dict
    config_file = tmp_path / "non_dict_config.yaml"
    config_file.write_text("- list item 1\n- list item 2")

    with pytest.raises(InvalidConfiguration):
        get_config(config_file)


# LLM-generated content at query #20
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "test_config.yaml"
    config_content = {
        'cookiecutters_dir': '~/test_cookiecutters/',
        'replay_dir': '~/test_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://custom.com/{0}'}
    }
    with open(config_file, 'w') as f:
        yaml.dump(config_content, f)

    result = get_config(config_file)
    assert result['cookiecutters_dir'] == os.path.expanduser('~/test_cookiecutters/')
    assert result['replay_dir'] == os.path.expanduser('~/test_replay/')
    assert result['default_context'] == {'key': 'value'}
    assert result['abbreviations'] == {**BUILTIN_ABBREVIATIONS, 'custom': 'https://custom.com/{0}'}

    # Test with a non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with an invalid YAML file
    invalid_yaml_file = tmp_path / "invalid.yaml"
    with open(invalid_yaml_file, 'w') as f:
        f.write("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with a YAML file that is not a dict
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    with open(non_dict_yaml_file, 'w') as f:
        f.write("- list item")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #21
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with valid YAML file
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text("""
cookiecutters_dir: /custom/cookiecutters/
replay_dir: /custom/replay/
default_context:
    key1: value1
abbreviations:
    custom: https://custom.com/{0}
""")
    result = get_config(config_file)
    assert result['cookiecutters_dir'] == '/custom/cookiecutters/'
    assert result['replay_dir'] == '/custom/replay/'
    assert result['default_context'] == {'key1': 'value1'}
    assert result['abbreviations']['custom'] == 'https://custom.com/{0}'
    assert result['abbreviations']['gh'] == BUILTIN_ABBREVIATIONS['gh']

    # Test with environment variable expansion
    config_file.write_text("""
cookiecutters_dir: $HOME/test_dir
replay_dir: ~/test_replay
""")
    result = get_config(config_file)
    assert result['cookiecutters_dir'] == os.path.expandvars('$HOME/test_dir')
    assert result['replay_dir'] == os.path.expanduser('~/test_replay')

    # Test with non-existent file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with invalid YAML
    config_file.write_text("invalid yaml content")
    with pytest.raises(InvalidConfiguration):
        get_config(config_file)

    # Test with non-dict YAML content
    config_file.write_text("not a dict")
    with pytest.raises(InvalidConfiguration):
        get_config(config_file)


# LLM-generated content at query #22
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "test_config.yaml"
    config_content = {
        'cookiecutters_dir': '~/test_cookiecutters/',
        'replay_dir': '~/test_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://custom.com/{0}'}
    }
    with open(config_file, 'w') as f:
        yaml.dump(config_content, f)

    result = get_config(config_file)
    assert result['cookiecutters_dir'] == os.path.expanduser('~/test_cookiecutters/')
    assert result['replay_dir'] == os.path.expanduser('~/test_replay/')
    assert result['default_context'] == collections.OrderedDict([('key', 'value')])
    assert result['abbreviations'] == {**BUILTIN_ABBREVIATIONS, 'custom': 'https://custom.com/{0}'}

    # Test with a non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with an invalid YAML file
    invalid_yaml_file = tmp_path / "invalid.yaml"
    with open(invalid_yaml_file, 'w') as f:
        f.write("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with a YAML file that is not a dict
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    with open(non_dict_yaml_file, 'w') as f:
        f.write("- list item")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #23
#--------------------------

```python
def test_get_config(mocker, tmp_path):
    # Test with a valid config file
    config_file = tmp_path / 'config.yaml'
    config_file.write_text('''
cookiecutters_dir: /custom/cookiecutters/
replay_dir: /custom/replay/
default_context:
    key: value
abbreviations:
    custom: https://custom.com/{0}
''')
    result = get_config(config_file)
    assert result['cookiecutters_dir'] == '/custom/cookiecutters/'
    assert result['replay_dir'] == '/custom/replay/'
    assert result['default_context'] == {'key': 'value'}
    assert result['abbreviations']['custom'] == 'https://custom.com/{0}'
    assert result['abbreviations']['gh'] == 'https://github.com/{0}.git'

    # Test with environment variable expansion
    config_file.write_text('''
cookiecutters_dir: $HOME/test/
replay_dir: ~/test/
''')
    result = get_config(config_file)
    assert result['cookiecutters_dir'] == os.path.expandvars('$HOME/test/')
    assert result['replay_dir'] == os.path.expanduser('~/test/')

    # Test with non-existent config file
    non_existent_file = tmp_path / 'non_existent.yaml'
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with invalid YAML
    config_file.write_text('invalid yaml content')
    with pytest.raises(InvalidConfiguration):
        get_config(config_file)

    # Test with non-dict YAML content
    config_file.write_text('- list item')
    with pytest.raises(InvalidConfiguration):
        get_config(config_file)

    # Test with empty YAML file
    config_file.write_text('')
    result = get_config(config_file)
    assert result['cookiecutters_dir'] == DEFAULT_CONFIG['cookiecutters_dir']
    assert result['replay_dir'] == DEFAULT_CONFIG['replay_dir']
    assert result['default_context'] == DEFAULT_CONFIG['default_context']
    assert result['abbreviations'] == DEFAULT_CONFIG['abbreviations']


# LLM-generated content at query #24
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / 'config.yaml'
    config_content = {
        'cookiecutters_dir': '~/test_cookiecutters/',
        'replay_dir': '~/test_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://custom.com/{0}'},
    }
    with open(config_file, 'w') as f:
        yaml.dump(config_content, f)

    result = get_config(config_file)
    assert result['cookiecutters_dir'] == os.path.expanduser('~/test_cookiecutters/')
    assert result['replay_dir'] == os.path.expanduser('~/test_replay/')
    assert result['default_context'] == {'key': 'value'}
    assert result['abbreviations'] == {**BUILTIN_ABBREVIATIONS, 'custom': 'https://custom.com/{0}'}

    # Test with a non-existent config file
    non_existent_file = tmp_path / 'non_existent.yaml'
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with an invalid YAML file
    invalid_yaml_file = tmp_path / 'invalid.yaml'
    with open(invalid_yaml_file, 'w') as f:
        f.write('invalid: yaml: content: [unclosed')
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with a YAML file that is not a dict
    non_dict_yaml_file = tmp_path / 'non_dict.yaml'
    with open(non_dict_yaml_file, 'w') as f:
        f.write('- list item 1\n- list item 2')
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #25
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / 'config.yaml'
    config_file.write_text('''
cookiecutters_dir: /custom/cookiecutters/
replay_dir: /custom/replay/
default_context:
    key: value
abbreviations:
    custom: https://custom.com/{0}
''')

    config = get_config(config_file)
    assert config['cookiecutters_dir'] == '/custom/cookiecutters/'
    assert config['replay_dir'] == '/custom/replay/'
    assert config['default_context'] == {'key': 'value'}
    assert config['abbreviations']['custom'] == 'https://custom.com/{0}'
    assert config['abbreviations']['gh'] == 'https://github.com/{0}.git'

    # Test with environment variable expansion
    config_file.write_text('''
cookiecutters_dir: $HOME/test/
replay_dir: ~/test/
''')
    config = get_config(config_file)
    assert config['cookiecutters_dir'] == os.path.expandvars('$HOME/test/')
    assert config['replay_dir'] == os.path.expanduser('~/test/')

    # Test with non-existent config file
    non_existent_file = tmp_path / 'non_existent.yaml'
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with invalid YAML
    invalid_yaml_file = tmp_path / 'invalid.yaml'
    invalid_yaml_file.write_text('invalid: yaml: content: [unclosed')
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with non-dict YAML content
    non_dict_yaml_file = tmp_path / 'non_dict.yaml'
    non_dict_yaml_file.write_text('- list item')
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #26
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "test_config.yaml"
    config_content = {
        'cookiecutters_dir': '~/test_cookiecutters/',
        'replay_dir': '~/test_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://custom.com/{0}'}
    }
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config_content, f)

    result = get_config(config_file)
    assert result['cookiecutters_dir'] == os.path.expanduser('~/test_cookiecutters/')
    assert result['replay_dir'] == os.path.expanduser('~/test_replay/')
    assert result['default_context'] == {'key': 'value'}
    assert result['abbreviations'] == {**BUILTIN_ABBREVIATIONS, 'custom': 'https://custom.com/{0}'}

    # Test with a non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with an invalid YAML file
    invalid_yaml_file = tmp_path / "invalid_yaml.yaml"
    with open(invalid_yaml_file, 'w', encoding='utf-8') as f:
        f.write("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with a YAML file that is not a dict
    non_dict_yaml_file = tmp_path / "non_dict_yaml.yaml"
    with open(non_dict_yaml_file, 'w', encoding='utf-8') as f:
        f.write("- list_item1\n- list_item2")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #27
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "test_config.yaml"
    config_content = {
        'cookiecutters_dir': '~/test_cookiecutters/',
        'replay_dir': '~/test_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://custom.com/{0}'}
    }
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config_content, f)

    config = get_config(config_file)
    assert config['cookiecutters_dir'] == os.path.expanduser('~/test_cookiecutters/')
    assert config['replay_dir'] == os.path.expanduser('~/test_replay/')
    assert config['default_context'] == {'key': 'value'}
    assert config['abbreviations'] == {**BUILTIN_ABBREVIATIONS, 'custom': 'https://custom.com/{0}'}

    # Test with a non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with an invalid YAML file
    invalid_yaml_file = tmp_path / "invalid.yaml"
    with open(invalid_yaml_file, 'w', encoding='utf-8') as f:
        f.write("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with a YAML file that doesn't have a dict at the top level
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    with open(non_dict_yaml_file, 'w', encoding='utf-8') as f:
        f.write("- list item")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #28
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(
        """
        cookiecutters_dir: /custom/cookiecutters/
        replay_dir: /custom/replay/
        default_context:
            key1: value1
        abbreviations:
            custom: https://custom.com/{0}
        """
    )

    result = get_config(config_file)
    assert result["cookiecutters_dir"] == "/custom/cookiecutters/"
    assert result["replay_dir"] == "/custom/replay/"
    assert result["default_context"] == {"key1": "value1"}
    assert result["abbreviations"]["custom"] == "https://custom.com/{0}"
    assert result["abbreviations"]["gh"] == "https://github.com/{0}.git"

    # Test with environment variable expansion
    config_file.write_text(
        """
        cookiecutters_dir: $HOME/test_cookiecutters/
        replay_dir: ~/test_replay/
        """
    )
    result = get_config(config_file)
    assert result["cookiecutters_dir"] == os.path.expandvars("$HOME/test_cookiecutters/")
    assert result["replay_dir"] == os.path.expanduser("~/test_replay/")

    # Test with non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with invalid YAML
    invalid_yaml_file = tmp_path / "invalid.yaml"
    invalid_yaml_file.write_text("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with non-dict YAML content
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    non_dict_yaml_file.write_text("- list item")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #29
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "test_config.yaml"
    config_content = {
        'cookiecutters_dir': '~/test_cookiecutters/',
        'replay_dir': '~/test_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://custom.com/{0}'},
    }
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config_content, f)

    result = get_config(config_file)

    assert result['cookiecutters_dir'] == os.path.expanduser('~/test_cookiecutters/')
    assert result['replay_dir'] == os.path.expanduser('~/test_replay/')
    assert result['default_context'] == {'key': 'value'}
    assert result['abbreviations'] == {**BUILTIN_ABBREVIATIONS, 'custom': 'https://custom.com/{0}'}

    # Test with a non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with an invalid YAML file
    invalid_yaml_file = tmp_path / "invalid.yaml"
    with open(invalid_yaml_file, 'w', encoding='utf-8') as f:
        f.write("invalid: yaml: content: [unclosed")

    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with a YAML file that is not a dict
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    with open(non_dict_yaml_file, 'w', encoding='utf-8') as f:
        f.write("- list item 1\n- list item 2")

    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #30
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "test_config.yaml"
    config_content = {
        "cookiecutters_dir": "~/test_cookiecutters/",
        "replay_dir": "~/test_replay/",
        "default_context": {"key": "value"},
        "abbreviations": {"custom": "https://custom.com/{0}"},
    }
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump(config_content, f)

    result = get_config(config_file)
    assert result["cookiecutters_dir"] == os.path.expanduser("~/test_cookiecutters/")
    assert result["replay_dir"] == os.path.expanduser("~/test_replay/")
    assert result["default_context"] == {"key": "value"}
    assert result["abbreviations"] == {**BUILTIN_ABBREVIATIONS, "custom": "https://custom.com/{0}"}

    # Test with a non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with an invalid YAML file
    invalid_yaml_file = tmp_path / "invalid.yaml"
    with open(invalid_yaml_file, "w", encoding="utf-8") as f:
        f.write("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with a YAML file that is not a dict
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    with open(non_dict_yaml_file, "w", encoding="utf-8") as f:
        f.write("- list item 1\n- list item 2")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #31
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "test_config.yaml"
    config_data = {
        'cookiecutters_dir': '~/test_cookiecutters/',
        'replay_dir': '~/test_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://custom.com/{0}'}
    }
    with open(config_file, 'w') as f:
        yaml.dump(config_data, f)

    result = get_config(config_file)

    assert result['cookiecutters_dir'] == os.path.expanduser('~/test_cookiecutters/')
    assert result['replay_dir'] == os.path.expanduser('~/test_replay/')
    assert result['default_context'] == {'key': 'value'}
    assert result['abbreviations'] == {**BUILTIN_ABBREVIATIONS, 'custom': 'https://custom.com/{0}'}

    # Test with a non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with an invalid YAML file
    invalid_yaml_file = tmp_path / "invalid.yaml"
    with open(invalid_yaml_file, 'w') as f:
        f.write("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with a YAML file that is not a dict
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    with open(non_dict_yaml_file, 'w') as f:
        f.write("- list_item1\n- list_item2")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #32
#--------------------------

```python
def test_get_config():
    # Test with a valid config file
    valid_config = {
        'cookiecutters_dir': '~/.test_cookiecutters/',
        'replay_dir': '~/.test_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://custom.com/{0}'}
    }
    config_file = 'test_config.yaml'
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(valid_config, f)

    config = get_config(config_file)
    assert config['cookiecutters_dir'] == os.path.expanduser('~/.test_cookiecutters/')
    assert config['replay_dir'] == os.path.expanduser('~/.test_replay/')
    assert config['default_context'] == {'key': 'value'}
    assert config['abbreviations'] == {**BUILTIN_ABBREVIATIONS, 'custom': 'https://custom.com/{0}'}

    os.remove(config_file)

    # Test with a non-existent config file
    with pytest.raises(ConfigDoesNotExistException):
        get_config('non_existent_config.yaml')

    # Test with an invalid YAML file
    invalid_yaml_file = 'invalid_config.yaml'
    with open(invalid_yaml_file, 'w', encoding='utf-8') as f:
        f.write('invalid: yaml: content: [')
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)
    os.remove(invalid_yaml_file)

    # Test with a YAML file that is not a dict
    non_dict_yaml_file = 'non_dict_config.yaml'
    with open(non_dict_yaml_file, 'w', encoding='utf-8') as f:
        f.write('- not a dict')
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)
    os.remove(non_dict_yaml_file)


# LLM-generated content at query #33
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "test_config.yaml"
    config_content = {
        'cookiecutters_dir': '~/test_cookiecutters/',
        'replay_dir': '~/test_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://custom.com/{0}'}
    }
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config_content, f)

    config = get_config(config_file)
    assert config['cookiecutters_dir'] == os.path.expanduser('~/test_cookiecutters/')
    assert config['replay_dir'] == os.path.expanduser('~/test_replay/')
    assert config['default_context'] == {'key': 'value'}
    assert config['abbreviations'] == {**BUILTIN_ABBREVIATIONS, 'custom': 'https://custom.com/{0}'}

    # Test with a non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with an invalid YAML file
    invalid_yaml_file = tmp_path / "invalid.yaml"
    with open(invalid_yaml_file, 'w', encoding='utf-8') as f:
        f.write("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with a YAML file that is not a dict
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    with open(non_dict_yaml_file, 'w', encoding='utf-8') as f:
        f.write("- list item 1\n- list item 2")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #34
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "test_config.yaml"
    config_content = {
        'cookiecutters_dir': '~/test_cookiecutters/',
        'replay_dir': '~/test_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://custom.com/{0}'}
    }
    with open(config_file, 'w') as f:
        yaml.dump(config_content, f)

    result = get_config(config_file)

    assert result['cookiecutters_dir'] == os.path.expanduser('~/test_cookiecutters/')
    assert result['replay_dir'] == os.path.expanduser('~/test_replay/')
    assert result['default_context'] == {'key': 'value'}
    assert result['abbreviations'] == {**BUILTIN_ABBREVIATIONS, 'custom': 'https://custom.com/{0}'}

    # Test with a non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with an invalid YAML file
    invalid_yaml_file = tmp_path / "invalid.yaml"
    with open(invalid_yaml_file, 'w') as f:
        f.write("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with a YAML file that is not a dict
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    with open(non_dict_yaml_file, 'w') as f:
        f.write("- list item 1\n- list item 2")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #35
#--------------------------

```python
def test_get_config():
    # Test with a valid config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("cookiecutters_dir: /test/dir\nreplay_dir: /test/replay\nabbreviations:\n  test: test_value")
        f.flush()
        config = get_config(f.name)
        assert config['cookiecutters_dir'] == '/test/dir'
        assert config['replay_dir'] == '/test/replay'
        assert config['abbreviations']['test'] == 'test_value'
        os.unlink(f.name)

    # Test with a non-existent config file
    with pytest.raises(ConfigDoesNotExistException):
        get_config('/non/existent/path.yaml')

    # Test with an invalid YAML file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("invalid: yaml: content: [")
        f.flush()
        with pytest.raises(InvalidConfiguration):
            get_config(f.name)
        os.unlink(f.name)

    # Test with a YAML file that is not a dict
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("- not a dict")
        f.flush()
        with pytest.raises(InvalidConfiguration):
            get_config(f.name)
        os.unlink(f.name)

    # Test path expansion
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("cookiecutters_dir: ~/test/dir\nreplay_dir: ~/test/replay")
        f.flush()
        config = get_config(f.name)
        assert config['cookiecutters_dir'] == os.path.expanduser('~/test/dir')
        assert config['replay_dir'] == os.path.expanduser('~/test/replay')
        os.unlink(f.name)


# LLM-generated content at query #36
#--------------------------

```python
def test_get_user_config():
    # Test default config
    config = get_user_config(default_config=True)
    assert config == DEFAULT_CONFIG

    # Test custom config dict
    custom_config = {'cookiecutters_dir': '/custom/dir'}
    config = get_user_config(default_config=custom_config)
    expected = merge_configs(DEFAULT_CONFIG, custom_config)
    assert config == expected

    # Test loading from custom config file
    with patch('builtins.open', mock_open(read_data='cookiecutters_dir: /test/dir')):
        with patch('os.path.exists', return_value=True):
            config = get_user_config(config_file='/test/config')
            assert config['cookiecutters_dir'] == '/test/dir'

    # Test loading from environment variable
    with patch.dict('os.environ', {'COOKIECUTTER_CONFIG': '/env/config'}):
        with patch('builtins.open', mock_open(read_data='cookiecutters_dir: /env/dir')):
            config = get_user_config()
            assert config['cookiecutters_dir'] == '/env/dir'

    # Test loading from default user config path
    with patch('os.path.exists', return_value=True):
        with patch('builtins.open', mock_open(read_data='cookiecutters_dir: /user/dir')):
            config = get_user_config()
            assert config['cookiecutters_dir'] == '/user/dir'

    # Test fallback to default config when no config file exists
    with patch('os.path.exists', return_value=False):
        config = get_user_config()
        assert config == DEFAULT_CONFIG


# LLM-generated content at query #37
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / 'config.yaml'
    config_file.write_text('''
cookiecutters_dir: /custom/cookiecutters/
replay_dir: /custom/replay/
default_context:
    key1: value1
abbreviations:
    custom: https://custom.com/{0}
''')

    config = get_config(config_file)
    assert config['cookiecutters_dir'] == '/custom/cookiecutters/'
    assert config['replay_dir'] == '/custom/replay/'
    assert config['default_context'] == {'key1': 'value1'}
    assert config['abbreviations']['custom'] == 'https://custom.com/{0}'
    assert config['abbreviations']['gh'] == 'https://github.com/{0}.git'

    # Test with environment variable expansion
    config_file.write_text('''
cookiecutters_dir: $HOME/test/
replay_dir: ~/replay/
''')
    config = get_config(config_file)
    assert config['cookiecutters_dir'] == os.path.expandvars('$HOME/test/')
    assert config['replay_dir'] == os.path.expanduser('~/replay/')

    # Test with non-existent config file
    non_existent_file = tmp_path / 'non_existent.yaml'
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with invalid YAML
    invalid_yaml_file = tmp_path / 'invalid.yaml'
    invalid_yaml_file.write_text('invalid: yaml: content: [unclosed')
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with non-dict YAML content
    non_dict_yaml_file = tmp_path / 'non_dict.yaml'
    non_dict_yaml_file.write_text('- list item 1\n- list item 2')
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #38
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "test_config.yaml"
    config_content = {
        'cookiecutters_dir': '~/test_cookiecutters/',
        'replay_dir': '~/test_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://custom.com/{0}'}
    }
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config_content, f)

    config = get_config(config_file)
    assert config['cookiecutters_dir'] == os.path.expanduser('~/test_cookiecutters/')
    assert config['replay_dir'] == os.path.expanduser('~/test_replay/')
    assert config['default_context'] == {'key': 'value'}
    assert config['abbreviations'] == {**BUILTIN_ABBREVIATIONS, 'custom': 'https://custom.com/{0}'}

    # Test with a non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with an invalid YAML file
    invalid_yaml_file = tmp_path / "invalid.yaml"
    with open(invalid_yaml_file, 'w', encoding='utf-8') as f:
        f.write("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with a YAML file that is not a dict
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    with open(non_dict_yaml_file, 'w', encoding='utf-8') as f:
        f.write("- list item 1\n- list item 2")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #39
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "test_config.yaml"
    config_content = {
        'cookiecutters_dir': '~/test_cookiecutters/',
        'replay_dir': '~/test_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://custom.com/{0}'}
    }
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config_content, f)

    config = get_config(config_file)
    assert config['cookiecutters_dir'] == os.path.expanduser('~/test_cookiecutters/')
    assert config['replay_dir'] == os.path.expanduser('~/test_replay/')
    assert config['default_context'] == {'key': 'value'}
    assert config['abbreviations'] == {**BUILTIN_ABBREVIATIONS, 'custom': 'https://custom.com/{0}'}

    # Test with a non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with an invalid YAML file
    invalid_yaml_file = tmp_path / "invalid.yaml"
    with open(invalid_yaml_file, 'w', encoding='utf-8') as f:
        f.write("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with a YAML file that is not a dict
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    with open(non_dict_yaml_file, 'w', encoding='utf-8') as f:
        f.write("- list item")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #40
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
        cookiecutters_dir: /custom/cookiecutters/
        replay_dir: /custom/replay/
        default_context:
            key: value
        abbreviations:
            custom: https://custom.com/{0}
        """
    )

    config = get_config(config_file)
    assert config["cookiecutters_dir"] == "/custom/cookiecutters/"
    assert config["replay_dir"] == "/custom/replay/"
    assert config["default_context"] == {"key": "value"}
    assert config["abbreviations"]["custom"] == "https://custom.com/{0}"
    assert config["abbreviations"]["gh"] == "https://github.com/{0}.git"

    # Test with environment variable expansion
    config_file.write_text(
        """
        cookiecutters_dir: $HOME/test/cookiecutters/
        replay_dir: ~/test/replay/
        """
    )
    config = get_config(config_file)
    assert config["cookiecutters_dir"] == os.path.expandvars("$HOME/test/cookiecutters/")
    assert config["replay_dir"] == os.path.expanduser("~/test/replay/")

    # Test with non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with invalid YAML
    invalid_yaml_file = tmp_path / "invalid.yaml"
    invalid_yaml_file.write_text("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with non-dict YAML
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    non_dict_yaml_file.write_text("- list item")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #41
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "test_config.yaml"
    config_content = {
        'cookiecutters_dir': '~/test_cookiecutters/',
        'replay_dir': '~/test_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://custom.com/{0}'}
    }
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config_content, f)

    result = get_config(config_file)
    assert result['cookiecutters_dir'] == os.path.expanduser('~/test_cookiecutters/')
    assert result['replay_dir'] == os.path.expanduser('~/test_replay/')
    assert result['default_context'] == {'key': 'value'}
    assert result['abbreviations'] == {**BUILTIN_ABBREVIATIONS, 'custom': 'https://custom.com/{0}'}

    # Test with a non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with an invalid YAML file
    invalid_yaml_file = tmp_path / "invalid.yaml"
    with open(invalid_yaml_file, 'w', encoding='utf-8') as f:
        f.write("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with a YAML file that is not a dict
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    with open(non_dict_yaml_file, 'w', encoding='utf-8') as f:
        f.write("- list item")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #42
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "test_config.yaml"
    config_content = {
        'cookiecutters_dir': '~/test_cookiecutters/',
        'replay_dir': '~/test_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://custom.com/{0}'}
    }
    with open(config_file, 'w') as f:
        yaml.dump(config_content, f)

    result = get_config(config_file)
    assert result['cookiecutters_dir'] == os.path.expanduser('~/test_cookiecutters/')
    assert result['replay_dir'] == os.path.expanduser('~/test_replay/')
    assert result['default_context'] == {'key': 'value'}
    assert result['abbreviations'] == {**BUILTIN_ABBREVIATIONS, 'custom': 'https://custom.com/{0}'}

    # Test with a non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with an invalid YAML file
    invalid_yaml_file = tmp_path / "invalid.yaml"
    with open(invalid_yaml_file, 'w') as f:
        f.write("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with a YAML file that is not a dict
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    with open(non_dict_yaml_file, 'w') as f:
        f.write("- list item")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #43
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "test_config.yaml"
    config_content = {
        'cookiecutters_dir': '~/test_cookiecutters/',
        'replay_dir': '~/test_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://custom.com/{0}'}
    }
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config_content, f)

    result = get_config(config_file)
    assert result['cookiecutters_dir'] == os.path.expanduser('~/test_cookiecutters/')
    assert result['replay_dir'] == os.path.expanduser('~/test_replay/')
    assert result['default_context'] == {'key': 'value'}
    assert result['abbreviations'] == {**BUILTIN_ABBREVIATIONS, 'custom': 'https://custom.com/{0}'}

    # Test with non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with invalid YAML
    invalid_yaml_file = tmp_path / "invalid.yaml"
    with open(invalid_yaml_file, 'w', encoding='utf-8') as f:
        f.write("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with non-dict YAML content
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    with open(non_dict_yaml_file, 'w', encoding='utf-8') as f:
        f.write("- list item")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #44
#--------------------------

```python
def test_get_user_config():
    # Test default config
    config = get_user_config(default_config=True)
    assert config == DEFAULT_CONFIG

    # Test custom config dict
    custom_config = {'cookiecutters_dir': '/custom/dir'}
    config = get_user_config(default_config=custom_config)
    expected = merge_configs(DEFAULT_CONFIG, custom_config)
    assert config == expected

    # Test loading from custom config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("cookiecutters_dir: /test/dir\n")
        f.write("replay_dir: /test/replay\n")
        f.write("default_context:\n")
        f.write("  key: value\n")
        f.flush()
        config = get_user_config(config_file=f.name)
        assert config['cookiecutters_dir'] == '/test/dir'
        assert config['replay_dir'] == '/test/replay'
        assert config['default_context'] == {'key': 'value'}
        os.unlink(f.name)

    # Test environment variable
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("cookiecutters_dir: /env/dir\n")
        f.flush()
        os.environ['COOKIECUTTER_CONFIG'] = f.name
        config = get_user_config()
        assert config['cookiecutters_dir'] == '/env/dir'
        del os.environ['COOKIECUTTER_CONFIG']
        os.unlink(f.name)

    # Test non-existent config file raises exception
    with pytest.raises(ConfigDoesNotExistException):
        get_user_config(config_file='/non/existent/file.yaml')

    # Test invalid YAML raises exception
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("invalid: yaml: content\n")
        f.write("- list item\n")
        f.flush()
        with pytest.raises(InvalidConfiguration):
            get_user_config(config_file=f.name)
        os.unlink(f.name)


# LLM-generated content at query #45
#--------------------------

```python
def test_get_user_config():
    # Test default config
    config = get_user_config(default_config=True)
    assert config == DEFAULT_CONFIG

    # Test custom config dict
    custom_config = {'cookiecutters_dir': '/custom/dir'}
    config = get_user_config(default_config=custom_config)
    expected = merge_configs(DEFAULT_CONFIG, custom_config)
    assert config == expected

    # Test loading from custom config file
    with patch('builtins.open', mock_open(read_data='cookiecutters_dir: /test/dir')):
        with patch('os.path.exists', return_value=True):
            config = get_user_config(config_file='/test/config')
            assert config['cookiecutters_dir'] == '/test/dir'

    # Test loading from environment variable
    with patch.dict('os.environ', {'COOKIECUTTER_CONFIG': '/env/config'}):
        with patch('builtins.open', mock_open(read_data='cookiecutters_dir: /env/dir')):
            config = get_user_config()
            assert config['cookiecutters_dir'] == '/env/dir'

    # Test loading from default user config path
    with patch('os.path.exists', return_value=True):
        with patch('builtins.open', mock_open(read_data='cookiecutters_dir: /user/dir')):
            config = get_user_config()
            assert config['cookiecutters_dir'] == '/user/dir'

    # Test fallback to default config
    with patch('os.path.exists', return_value=False):
        with patch.dict('os.environ', {}, clear=True):
            config = get_user_config()
            assert config == DEFAULT_CONFIG


# LLM-generated content at query #46
#--------------------------

```python
def test_get_user_config():
    # Test default config
    config = get_user_config(default_config=True)
    assert config == DEFAULT_CONFIG

    # Test custom default config
    custom_default = {'cookiecutters_dir': '/custom/dir'}
    config = get_user_config(default_config=custom_default)
    expected = merge_configs(DEFAULT_CONFIG, custom_default)
    assert config == expected

    # Test loading from custom config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write('cookiecutters_dir: /test/dir\nabbreviations:\n  custom: "test"')
        f.flush()
        config = get_user_config(config_file=f.name)
        assert config['cookiecutters_dir'] == '/test/dir'
        assert config['abbreviations']['custom'] == 'test'
        os.unlink(f.name)

    # Test COOKIECUTTER_CONFIG environment variable
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write('replay_dir: /env/dir')
        f.flush()
        os.environ['COOKIECUTTER_CONFIG'] = f.name
        config = get_user_config()
        assert config['replay_dir'] == '/env/dir'
        del os.environ['COOKIECUTTER_CONFIG']
        os.unlink(f.name)

    # Test loading from default user config path
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write('default_context:\n  key: value')
        f.flush()
        os.rename(f.name, USER_CONFIG_PATH)
        config = get_user_config()
        assert config['default_context']['key'] == 'value'
        os.unlink(USER_CONFIG_PATH)


# LLM-generated content at query #47
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "test_config.yaml"
    config_content = {
        'cookiecutters_dir': '~/custom_cookiecutters/',
        'replay_dir': '~/custom_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://custom.com/{0}'}
    }
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config_content, f)

    config = get_config(config_file)
    assert config['cookiecutters_dir'] == os.path.expanduser('~/custom_cookiecutters/')
    assert config['replay_dir'] == os.path.expanduser('~/custom_replay/')
    assert config['default_context'] == {'key': 'value'}
    assert config['abbreviations'] == {**BUILTIN_ABBREVIATIONS, 'custom': 'https://custom.com/{0}'}

    # Test with a non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with an invalid YAML file
    invalid_yaml_file = tmp_path / "invalid.yaml"
    with open(invalid_yaml_file, 'w', encoding='utf-8') as f:
        f.write("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with a YAML file that is not a dict
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    with open(non_dict_yaml_file, 'w', encoding='utf-8') as f:
        f.write("- list item")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #48
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "config.yaml"
    config_content = {
        'cookiecutters_dir': '~/test_cookiecutters/',
        'replay_dir': '~/test_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://custom.com/{0}'}
    }
    config_file.write_text(yaml.dump(config_content))

    result = get_config(config_file)

    assert result['cookiecutters_dir'] == os.path.expanduser('~/test_cookiecutters/')
    assert result['replay_dir'] == os.path.expanduser('~/test_replay/')
    assert result['default_context'] == {'key': 'value'}
    assert result['abbreviations'] == {**BUILTIN_ABBREVIATIONS, 'custom': 'https://custom.com/{0}'}

    # Test with a non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with an invalid YAML file
    invalid_yaml_file = tmp_path / "invalid.yaml"
    invalid_yaml_file.write_text("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with a YAML file that is not a dict
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    non_dict_yaml_file.write_text("- list item")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #49
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "test_config.yaml"
    config_content = {
        "cookiecutters_dir": "~/test_cookiecutters/",
        "replay_dir": "~/test_replay/",
        "default_context": {"key": "value"},
        "abbreviations": {"custom": "https://custom.com/{0}"}
    }
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump(config_content, f)

    result = get_config(config_file)
    assert result["cookiecutters_dir"] == os.path.expanduser("~/test_cookiecutters/")
    assert result["replay_dir"] == os.path.expanduser("~/test_replay/")
    assert result["default_context"] == {"key": "value"}
    assert result["abbreviations"] == {**BUILTIN_ABBREVIATIONS, "custom": "https://custom.com/{0}"}

    # Test with a non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with an invalid YAML file
    invalid_yaml_file = tmp_path / "invalid.yaml"
    with open(invalid_yaml_file, "w", encoding="utf-8") as f:
        f.write("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with a YAML file that is not a dict
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    with open(non_dict_yaml_file, "w", encoding="utf-8") as f:
        f.write("- list item 1\n- list item 2")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #50
#--------------------------

```python
def test_get_user_config():
    # Test default config
    config = get_user_config(default_config=True)
    assert config == DEFAULT_CONFIG

    # Test custom config dict
    custom_config = {'cookiecutters_dir': '/custom/dir'}
    config = get_user_config(default_config=custom_config)
    expected = merge_configs(DEFAULT_CONFIG, custom_config)
    assert config == expected

    # Test loading from custom config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump({'cookiecutters_dir': '/test/dir'}, f)
        config_file = f.name

    try:
        config = get_user_config(config_file=config_file)
        assert config['cookiecutters_dir'] == '/test/dir'
    finally:
        os.unlink(config_file)

    # Test COOKIECUTTER_CONFIG environment variable
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump({'replay_dir': '/env/dir'}, f)
        env_config_file = f.name

    try:
        os.environ['COOKIECUTTER_CONFIG'] = env_config_file
        config = get_user_config()
        assert config['replay_dir'] == '/env/dir'
    finally:
        del os.environ['COOKIECUTTER_CONFIG']
        os.unlink(env_config_file)

    # Test loading from default user config path
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump({'default_context': {'key': 'value'}}, f)
        user_config_file = f.name

    try:
        with patch('cookiecutter.config.USER_CONFIG_PATH', user_config_file):
            config = get_user_config()
            assert config['default_context'] == {'key': 'value'}
    finally:
        os.unlink(user_config_file)

    # Test non-existent config file raises exception
    with pytest.raises(ConfigDoesNotExistException):
        get_user_config(config_file='/non/existent/file.yaml')


# LLM-generated content at query #51
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / 'test_config.yaml'
    config_content = {
        'cookiecutters_dir': '~/test_cookiecutters/',
        'replay_dir': '~/test_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://custom.com/{0}'}
    }
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config_content, f)

    result = get_config(config_file)
    assert result['cookiecutters_dir'] == os.path.expanduser('~/test_cookiecutters/')
    assert result['replay_dir'] == os.path.expanduser('~/test_replay/')
    assert result['default_context'] == {'key': 'value'}
    assert result['abbreviations'] == {**BUILTIN_ABBREVIATIONS, 'custom': 'https://custom.com/{0}'}

    # Test with a non-existent config file
    non_existent_file = tmp_path / 'non_existent.yaml'
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with an invalid YAML file
    invalid_yaml_file = tmp_path / 'invalid.yaml'
    with open(invalid_yaml_file, 'w', encoding='utf-8') as f:
        f.write('invalid: yaml: content: [')
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with a YAML file that is not a dict
    non_dict_yaml_file = tmp_path / 'non_dict.yaml'
    with open(non_dict_yaml_file, 'w', encoding='utf-8') as f:
        f.write('- list item')
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #52
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "test_config.yaml"
    config_content = {
        'cookiecutters_dir': '~/test_cookiecutters/',
        'replay_dir': '~/test_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://custom.com/{0}'}
    }
    with open(config_file, 'w') as f:
        yaml.dump(config_content, f)

    result = get_config(config_file)
    assert result['cookiecutters_dir'] == os.path.expanduser('~/test_cookiecutters/')
    assert result['replay_dir'] == os.path.expanduser('~/test_replay/')
    assert result['default_context'] == {'key': 'value'}
    assert result['abbreviations'] == {**BUILTIN_ABBREVIATIONS, 'custom': 'https://custom.com/{0}'}

    # Test with a non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with an invalid YAML file
    invalid_yaml_file = tmp_path / "invalid.yaml"
    with open(invalid_yaml_file, 'w') as f:
        f.write("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with a YAML file that is not a dict
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    with open(non_dict_yaml_file, 'w') as f:
        f.write("- not a dict")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #53
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "test_config.yaml"
    config_content = {
        'cookiecutters_dir': '~/test_cookiecutters/',
        'replay_dir': '~/test_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://custom.com/{0}'}
    }
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config_content, f)

    result = get_config(config_file)
    assert result['cookiecutters_dir'] == os.path.expanduser('~/test_cookiecutters/')
    assert result['replay_dir'] == os.path.expanduser('~/test_replay/')
    assert result['default_context'] == {'key': 'value'}
    assert result['abbreviations'] == {**BUILTIN_ABBREVIATIONS, 'custom': 'https://custom.com/{0}'}

    # Test with a non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with an invalid YAML file
    invalid_yaml_file = tmp_path / "invalid.yaml"
    with open(invalid_yaml_file, 'w', encoding='utf-8') as f:
        f.write("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with a YAML file that is not a dict
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    with open(non_dict_yaml_file, 'w', encoding='utf-8') as f:
        f.write("- list item")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #54
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "test_config.yaml"
    config_content = {
        'cookiecutters_dir': '~/test_cookiecutters/',
        'replay_dir': '~/test_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://custom.com/{0}'}
    }
    with open(config_file, 'w') as f:
        yaml.dump(config_content, f)

    result = get_config(config_file)
    assert result['cookiecutters_dir'] == os.path.expanduser('~/test_cookiecutters/')
    assert result['replay_dir'] == os.path.expanduser('~/test_replay/')
    assert result['default_context'] == {'key': 'value'}
    assert result['abbreviations'] == {**BUILTIN_ABBREVIATIONS, 'custom': 'https://custom.com/{0}'}

    # Test with a non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with an invalid YAML file
    invalid_yaml_file = tmp_path / "invalid.yaml"
    with open(invalid_yaml_file, 'w') as f:
        f.write("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with a YAML file that is not a dict
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    with open(non_dict_yaml_file, 'w') as f:
        f.write("- list_item1\n- list_item2")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #55
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(
        """
cookiecutters_dir: /custom/cookiecutters/
replay_dir: /custom/replay/
default_context:
    key1: value1
abbreviations:
    custom: https://custom.com/{0}
"""
    )

    result = get_config(config_file)

    assert result["cookiecutters_dir"] == "/custom/cookiecutters/"
    assert result["replay_dir"] == "/custom/replay/"
    assert result["default_context"] == {"key1": "value1"}
    assert result["abbreviations"] == {
        "gh": "https://github.com/{0}.git",
        "gl": "https://gitlab.com/{0}.git",
        "bb": "https://bitbucket.org/{0}",
        "custom": "https://custom.com/{0}",
    }

    # Test with environment variable expansion
    config_file.write_text(
        """
cookiecutters_dir: $HOME/test_cookiecutters/
replay_dir: ~/test_replay/
"""
    )
    result = get_config(config_file)
    assert result["cookiecutters_dir"] == os.path.expandvars("$HOME/test_cookiecutters/")
    assert result["replay_dir"] == os.path.expanduser("~/test_replay/")

    # Test with non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with invalid YAML
    invalid_yaml_file = tmp_path / "invalid.yaml"
    invalid_yaml_file.write_text("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with non-dict YAML content
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    non_dict_yaml_file.write_text("- list item 1\n- list item 2")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #56
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / 'config.yaml'
    config_file.write_text('''
cookiecutters_dir: /custom/cookiecutters/
replay_dir: /custom/replay/
default_context:
    key: value
abbreviations:
    custom: https://custom.com/{0}
''')

    config = get_config(config_file)
    assert config['cookiecutters_dir'] == '/custom/cookiecutters/'
    assert config['replay_dir'] == '/custom/replay/'
    assert config['default_context'] == {'key': 'value'}
    assert config['abbreviations']['custom'] == 'https://custom.com/{0}'
    assert config['abbreviations']['gh'] == 'https://github.com/{0}.git'

    # Test with environment variable expansion
    config_file.write_text('''
cookiecutters_dir: $HOME/test/cookiecutters/
replay_dir: ~/test/replay/
''')
    config = get_config(config_file)
    assert config['cookiecutters_dir'] == os.path.expandvars('$HOME/test/cookiecutters/')
    assert config['replay_dir'] == os.path.expanduser('~/test/replay/')

    # Test with non-existent config file
    non_existent_file = tmp_path / 'non_existent.yaml'
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with invalid YAML
    invalid_yaml_file = tmp_path / 'invalid.yaml'
    invalid_yaml_file.write_text('invalid: yaml: content: [unclosed')
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with non-dict YAML content
    non_dict_yaml_file = tmp_path / 'non_dict.yaml'
    non_dict_yaml_file.write_text('- list item')
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #57
#--------------------------

```python
def test_get_config():
    # Test with a valid config file
    valid_config = {
        'cookiecutters_dir': '~/.test_cookiecutters/',
        'replay_dir': '~/.test_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'test': 'https://test.com/{0}'}
    }
    with open('test_config.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(valid_config, f)

    config = get_config('test_config.yaml')
    assert config['cookiecutters_dir'] == os.path.expanduser('~/.test_cookiecutters/')
    assert config['replay_dir'] == os.path.expanduser('~/.test_replay/')
    assert config['default_context'] == {'key': 'value'}
    assert config['abbreviations'] == {**BUILTIN_ABBREVIATIONS, 'test': 'https://test.com/{0}'}

    # Test with a non-existent config file
    with pytest.raises(ConfigDoesNotExistException):
        get_config('non_existent_config.yaml')

    # Test with an invalid YAML file
    with open('invalid_config.yaml', 'w', encoding='utf-8') as f:
        f.write('invalid yaml: [')
    with pytest.raises(InvalidConfiguration):
        get_config('invalid_config.yaml')

    # Test with a YAML file that is not a dict
    with open('non_dict_config.yaml', 'w', encoding='utf-8') as f:
        f.write('- list item')
    with pytest.raises(InvalidConfiguration):
        get_config('non_dict_config.yaml')


# LLM-generated content at query #58
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "test_config.yaml"
    config_content = {
        'cookiecutters_dir': '~/test_cookiecutters/',
        'replay_dir': '~/test_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://custom.com/{0}'}
    }
    with open(config_file, 'w') as f:
        yaml.dump(config_content, f)

    result = get_config(config_file)
    assert result['cookiecutters_dir'] == os.path.expanduser('~/test_cookiecutters/')
    assert result['replay_dir'] == os.path.expanduser('~/test_replay/')
    assert result['default_context'] == {'key': 'value'}
    assert result['abbreviations'] == {**BUILTIN_ABBREVIATIONS, 'custom': 'https://custom.com/{0}'}

    # Test with a non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with an invalid YAML file
    invalid_yaml_file = tmp_path / "invalid.yaml"
    with open(invalid_yaml_file, 'w') as f:
        f.write("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with a YAML file that is not a dict
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    with open(non_dict_yaml_file, 'w') as f:
        f.write("- list item")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #59
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "test_config.yaml"
    config_content = {
        'cookiecutters_dir': '~/test_cookiecutters/',
        'replay_dir': '~/test_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://custom.com/{0}'}
    }
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config_content, f)

    result = get_config(config_file)
    assert result['cookiecutters_dir'] == os.path.expanduser('~/test_cookiecutters/')
    assert result['replay_dir'] == os.path.expanduser('~/test_replay/')
    assert result['default_context'] == {'key': 'value'}
    assert result['abbreviations'] == {**BUILTIN_ABBREVIATIONS, 'custom': 'https://custom.com/{0}'}

    # Test with a non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with an invalid YAML file
    invalid_yaml_file = tmp_path / "invalid.yaml"
    with open(invalid_yaml_file, 'w', encoding='utf-8') as f:
        f.write("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with a YAML file that is not a dict
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    with open(non_dict_yaml_file, 'w', encoding='utf-8') as f:
        f.write("- list item")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #60
#--------------------------

```python
def test_get_user_config():
    # Test default config
    config = get_user_config(default_config=True)
    assert config == DEFAULT_CONFIG

    # Test custom config dict
    custom_config = {'cookiecutters_dir': '/custom/dir'}
    config = get_user_config(default_config=custom_config)
    expected = merge_configs(DEFAULT_CONFIG, custom_config)
    assert config == expected

    # Test loading from custom config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write('cookiecutters_dir: /test/dir\n')
        f.flush()
        config = get_user_config(config_file=f.name)
        assert config['cookiecutters_dir'] == '/test/dir'
        os.unlink(f.name)

    # Test environment variable config
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write('cookiecutters_dir: /env/dir\n')
        f.flush()
        os.environ['COOKIECUTTER_CONFIG'] = f.name
        config = get_user_config()
        assert config['cookiecutters_dir'] == '/env/dir'
        del os.environ['COOKIECUTTER_CONFIG']
        os.unlink(f.name)

    # Test loading from default user config path
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write('cookiecutters_dir: /user/dir\n')
        f.flush()
        original_user_config = USER_CONFIG_PATH
        USER_CONFIG_PATH = f.name
        config = get_user_config()
        assert config['cookiecutters_dir'] == '/user/dir'
        USER_CONFIG_PATH = original_user_config
        os.unlink(f.name)

    # Test fallback to default config
    with patch('os.path.exists', return_value=False):
        config = get_user_config()
        assert config == DEFAULT_CONFIG


# LLM-generated content at query #61
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "test_config.yaml"
    config_content = {
        'cookiecutters_dir': '~/test_cookiecutters/',
        'replay_dir': '~/test_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://custom.com/{0}'}
    }
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config_content, f)

    result = get_config(config_file)
    assert result['cookiecutters_dir'] == os.path.expanduser('~/test_cookiecutters/')
    assert result['replay_dir'] == os.path.expanduser('~/test_replay/')
    assert result['default_context'] == {'key': 'value'}
    assert result['abbreviations'] == {**BUILTIN_ABBREVIATIONS, **{'custom': 'https://custom.com/{0}'}}

    # Test with a non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with an invalid YAML file
    invalid_yaml_file = tmp_path / "invalid.yaml"
    with open(invalid_yaml_file, 'w', encoding='utf-8') as f:
        f.write("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with a YAML file that is not a dict
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    with open(non_dict_yaml_file, 'w', encoding='utf-8') as f:
        f.write("- list item")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #62
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "test_config.yaml"
    config_content = {
        'cookiecutters_dir': '~/test_cookiecutters/',
        'replay_dir': '~/test_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://custom.com/{0}'}
    }
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config_content, f)

    result = get_config(config_file)
    assert result['cookiecutters_dir'] == os.path.expanduser('~/test_cookiecutters/')
    assert result['replay_dir'] == os.path.expanduser('~/test_replay/')
    assert result['default_context'] == {'key': 'value'}
    assert result['abbreviations'] == {**BUILTIN_ABBREVIATIONS, 'custom': 'https://custom.com/{0}'}

    # Test with a non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with an invalid YAML file
    invalid_yaml_file = tmp_path / "invalid.yaml"
    with open(invalid_yaml_file, 'w', encoding='utf-8') as f:
        f.write("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with a YAML file that is not a dict
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    with open(non_dict_yaml_file, 'w', encoding='utf-8') as f:
        f.write("- list item")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #63
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text("""
cookiecutters_dir: /custom/cookiecutters/
replay_dir: /custom/replay/
default_context:
    key: value
abbreviations:
    custom: https://custom.com/{0}
""")
    config = get_config(config_file)
    assert config['cookiecutters_dir'] == '/custom/cookiecutters/'
    assert config['replay_dir'] == '/custom/replay/'
    assert config['default_context'] == {'key': 'value'}
    assert config['abbreviations']['custom'] == 'https://custom.com/{0}'
    assert config['abbreviations']['gh'] == 'https://github.com/{0}.git'

    # Test with environment variable expansion
    config_file.write_text("""
cookiecutters_dir: $HOME/test/
replay_dir: ~/replay/
""")
    config = get_config(config_file)
    assert config['cookiecutters_dir'] == os.path.expandvars('$HOME/test/')
    assert config['replay_dir'] == os.path.expanduser('~/replay/')

    # Test with non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with invalid YAML
    invalid_yaml_file = tmp_path / "invalid.yaml"
    invalid_yaml_file.write_text("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with non-dict YAML content
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    non_dict_yaml_file.write_text("- list item")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #64
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "test_config.yaml"
    config_content = {
        'cookiecutters_dir': '~/test_cookiecutters/',
        'replay_dir': '~/test_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://custom.com/{0}'}
    }
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config_content, f)

    result = get_config(config_file)
    assert result['cookiecutters_dir'] == os.path.expanduser('~/test_cookiecutters/')
    assert result['replay_dir'] == os.path.expanduser('~/test_replay/')
    assert result['default_context'] == {'key': 'value'}
    assert result['abbreviations'] == {**BUILTIN_ABBREVIATIONS, 'custom': 'https://custom.com/{0}'}

    # Test with a non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with an invalid YAML file
    invalid_yaml_file = tmp_path / "invalid.yaml"
    with open(invalid_yaml_file, 'w', encoding='utf-8') as f:
        f.write("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with a YAML file that is not a dict
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    with open(non_dict_yaml_file, 'w', encoding='utf-8') as f:
        f.write("- list item 1\n- list item 2")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #65
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "test_config.yaml"
    config_content = {
        "cookiecutters_dir": "~/test_cookiecutters/",
        "replay_dir": "~/test_replay/",
        "default_context": {"key": "value"},
        "abbreviations": {"custom": "https://custom.com/{0}"}
    }
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump(config_content, f)

    result = get_config(config_file)
    assert result["cookiecutters_dir"] == os.path.expanduser("~/test_cookiecutters/")
    assert result["replay_dir"] == os.path.expanduser("~/test_replay/")
    assert result["default_context"] == {"key": "value"}
    assert result["abbreviations"] == {**BUILTIN_ABBREVIATIONS, "custom": "https://custom.com/{0}"}

    # Test with a non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with an invalid YAML file
    invalid_yaml_file = tmp_path / "invalid.yaml"
    with open(invalid_yaml_file, "w", encoding="utf-8") as f:
        f.write("invalid: yaml: content: [unclosed")

    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with a YAML file that is not a dict
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    with open(non_dict_yaml_file, "w", encoding="utf-8") as f:
        f.write("- list_item1\n- list_item2")

    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #66
#--------------------------

```python
def test_get_user_config():
    # Test default config
    config = get_user_config(default_config=True)
    assert config == DEFAULT_CONFIG

    # Test custom default config
    custom_default = {'cookiecutters_dir': '/custom/dir'}
    config = get_user_config(default_config=custom_default)
    expected = merge_configs(DEFAULT_CONFIG, custom_default)
    assert config == expected

    # Test loading from custom config file
    with pytest.raises(ConfigDoesNotExistException):
        get_user_config(config_file='/nonexistent/path')

    # Test loading from environment variable
    with pytest.raises(ConfigDoesNotExistException):
        os.environ['COOKIECUTTER_CONFIG'] = '/nonexistent/path'
        get_user_config()

    # Test loading from default user config path
    with pytest.raises(ConfigDoesNotExistException):
        get_user_config()

    # Test loading from existing config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump({'cookiecutters_dir': '/test/dir'}, f)
        f.flush()
        config = get_user_config(config_file=f.name)
        assert config['cookiecutters_dir'] == '/test/dir'
        os.unlink(f.name)

    # Test loading from environment variable with existing file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump({'replay_dir': '/test/replay'}, f)
        f.flush()
        os.environ['COOKIECUTTER_CONFIG'] = f.name
        config = get_user_config()
        assert config['replay_dir'] == '/test/replay'
        os.unlink(f.name)
        del os.environ['COOKIECUTTER_CONFIG']


# LLM-generated content at query #67
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "test_config.yaml"
    config_content = {
        'cookiecutters_dir': '~/test_cookiecutters/',
        'replay_dir': '~/test_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://custom.com/{0}'}
    }
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config_content, f)

    result = get_config(config_file)

    assert result['cookiecutters_dir'] == os.path.expanduser('~/test_cookiecutters/')
    assert result['replay_dir'] == os.path.expanduser('~/test_replay/')
    assert result['default_context'] == {'key': 'value'}
    assert result['abbreviations'] == {**BUILTIN_ABBREVIATIONS, 'custom': 'https://custom.com/{0}'}

    # Test with a non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with an invalid YAML file
    invalid_yaml_file = tmp_path / "invalid.yaml"
    with open(invalid_yaml_file, 'w', encoding='utf-8') as f:
        f.write("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with a YAML file that is not a dict
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    with open(non_dict_yaml_file, 'w', encoding='utf-8') as f:
        f.write("- list item")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #68
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "test_config.yaml"
    config_content = {
        "cookiecutters_dir": "~/test_cookiecutters/",
        "replay_dir": "~/test_replay/",
        "default_context": {"key": "value"},
        "abbreviations": {"custom": "https://custom.com/{0}"}
    }
    config_file.write_text(yaml.dump(config_content))

    result = get_config(config_file)
    assert result["cookiecutters_dir"] == os.path.expanduser("~/test_cookiecutters/")
    assert result["replay_dir"] == os.path.expanduser("~/test_replay/")
    assert result["default_context"] == {"key": "value"}
    assert result["abbreviations"] == {**BUILTIN_ABBREVIATIONS, "custom": "https://custom.com/{0}"}

    # Test with a non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with an invalid YAML file
    invalid_yaml_file = tmp_path / "invalid.yaml"
    invalid_yaml_file.write_text("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with a YAML file that doesn't have a dict at the top level
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    non_dict_yaml_file.write_text("- list item")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #69
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "test_config.yaml"
    config_content = {
        'cookiecutters_dir': '~/test_cookiecutters/',
        'replay_dir': '~/test_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://custom.com/{0}'}
    }
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config_content, f)

    config = get_config(config_file)
    assert config['cookiecutters_dir'] == os.path.expanduser('~/test_cookiecutters/')
    assert config['replay_dir'] == os.path.expanduser('~/test_replay/')
    assert config['default_context'] == {'key': 'value'}
    assert config['abbreviations'] == {**BUILTIN_ABBREVIATIONS, 'custom': 'https://custom.com/{0}'}

    # Test with a non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with an invalid YAML file
    invalid_yaml_file = tmp_path / "invalid.yaml"
    with open(invalid_yaml_file, 'w', encoding='utf-8') as f:
        f.write("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with a YAML file that is not a dict
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    with open(non_dict_yaml_file, 'w', encoding='utf-8') as f:
        f.write("- list item")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #70
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "test_config.yaml"
    config_content = {
        'cookiecutters_dir': '~/test_dir/',
        'replay_dir': '~/test_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://custom.com/{0}'}
    }
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config_content, f)

    config = get_config(config_file)
    assert config['cookiecutters_dir'] == os.path.expanduser('~/test_dir/')
    assert config['replay_dir'] == os.path.expanduser('~/test_replay/')
    assert config['default_context'] == {'key': 'value'}
    assert config['abbreviations'] == {**BUILTIN_ABBREVIATIONS, 'custom': 'https://custom.com/{0}'}

    # Test with a non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with an invalid YAML file
    invalid_yaml_file = tmp_path / "invalid.yaml"
    with open(invalid_yaml_file, 'w', encoding='utf-8') as f:
        f.write("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with a YAML file that is not a dict
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    with open(non_dict_yaml_file, 'w', encoding='utf-8') as f:
        f.write("- list item")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #71
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "test_config.yaml"
    config_content = {
        'cookiecutters_dir': '~/test_cookiecutters/',
        'replay_dir': '~/test_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://custom.com/{0}'}
    }
    with open(config_file, 'w') as f:
        yaml.dump(config_content, f)

    result = get_config(config_file)
    assert result['cookiecutters_dir'] == os.path.expanduser('~/test_cookiecutters/')
    assert result['replay_dir'] == os.path.expanduser('~/test_replay/')
    assert result['default_context'] == {'key': 'value'}
    assert result['abbreviations'] == {**BUILTIN_ABBREVIATIONS, 'custom': 'https://custom.com/{0}'}

    # Test with a non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with an invalid YAML file
    invalid_file = tmp_path / "invalid.yaml"
    with open(invalid_file, 'w') as f:
        f.write("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_file)

    # Test with a YAML file that is not a dict
    non_dict_file = tmp_path / "non_dict.yaml"
    with open(non_dict_file, 'w') as f:
        f.write("- not a dict")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_file)


# LLM-generated content at query #72
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "test_config.yaml"
    config_content = {
        'cookiecutters_dir': '~/test_cookiecutters/',
        'replay_dir': '~/test_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://custom.com/{0}'}
    }
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config_content, f)

    result = get_config(config_file)
    assert result['cookiecutters_dir'] == os.path.expanduser('~/test_cookiecutters/')
    assert result['replay_dir'] == os.path.expanduser('~/test_replay/')
    assert result['default_context'] == {'key': 'value'}
    assert result['abbreviations'] == {**BUILTIN_ABBREVIATIONS, 'custom': 'https://custom.com/{0}'}

    # Test with a non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with an invalid YAML file
    invalid_yaml_file = tmp_path / "invalid.yaml"
    with open(invalid_yaml_file, 'w', encoding='utf-8') as f:
        f.write("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with a YAML file that is not a dict
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    with open(non_dict_yaml_file, 'w', encoding='utf-8') as f:
        f.write("- list_item1\n- list_item2")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


