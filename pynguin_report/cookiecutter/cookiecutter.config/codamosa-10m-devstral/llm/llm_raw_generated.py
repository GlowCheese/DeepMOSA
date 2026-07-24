####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
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

    # Test with non-existent config file
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with invalid YAML
    invalid_yaml_file = tmp_path / "invalid.yaml"
    with open(invalid_yaml_file, 'w') as f:
        f.write("invalid: yaml: content: [unclosed")

    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with non-dict YAML content
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    with open(non_dict_yaml_file, 'w') as f:
        f.write("- a list\n- not a dict")

    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #2
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "test_config.yaml"
    config_content = {
        "cookiecutters_dir": "~/test_cookiecutters/",
        "replay_dir": "~/test_replay/",
        "default_context": {"key": "value"},
        "abbreviations": {"test": "test_url"},
    }
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump(config_content, f)

    result = get_config(config_file)
    assert result["cookiecutters_dir"] == os.path.expanduser("~/test_cookiecutters/")
    assert result["replay_dir"] == os.path.expanduser("~/test_replay/")
    assert result["default_context"] == collections.OrderedDict([("key", "value")])
    assert result["abbreviations"] == {**BUILTIN_ABBREVIATIONS, "test": "test_url"}

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


# LLM-generated content at query #3
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / 'config.yaml'
    config_content = {
        'cookiecutters_dir': '~/test_dir/',
        'replay_dir': '~/test_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://custom.com/{0}'}
    }
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config_content, f)

    result = get_config(config_file)
    assert result['cookiecutters_dir'] == os.path.expanduser('~/test_dir/')
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
        f.write('- list item 1\n- list item 2')
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
    invalid_file = tmp_path / "invalid.yaml"
    with open(invalid_file, 'w', encoding='utf-8') as f:
        f.write("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_file)

    # Test with a YAML file that is not a dict
    non_dict_file = tmp_path / "non_dict.yaml"
    with open(non_dict_file, 'w', encoding='utf-8') as f:
        f.write("- not a dict")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_file)


# LLM-generated content at query #5
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


# LLM-generated content at query #6
#--------------------------

```python
def test_get_user_config():
    # Test default config
    config = get_user_config(default_config=True)
    assert config == DEFAULT_CONFIG

    # Test custom default config
    custom_default = {'cookiecutters_dir': '/custom/path'}
    config = get_user_config(default_config=custom_default)
    expected = merge_configs(DEFAULT_CONFIG, custom_default)
    assert config == expected

    # Test loading from custom config file
    with patch('builtins.open', mock_open(read_data='cookiecutters_dir: /test/path')):
        with patch('os.path.exists', return_value=True):
            config = get_user_config(config_file='/test/config')
            assert config['cookiecutters_dir'] == '/test/path'

    # Test loading from environment variable
    with patch.dict('os.environ', {'COOKIECUTTER_CONFIG': '/env/config'}):
        with patch('builtins.open', mock_open(read_data='cookiecutters_dir: /env/path')):
            with patch('os.path.exists', return_value=True):
                config = get_user_config()
                assert config['cookiecutters_dir'] == '/env/path'

    # Test loading from default user config path
    with patch('os.path.exists', return_value=True):
        with patch('builtins.open', mock_open(read_data='cookiecutters_dir: /user/path')):
            config = get_user_config()
            assert config['cookiecutters_dir'] == '/user/path'

    # Test fallback to default config when no config file exists
    with patch('os.path.exists', return_value=False):
        config = get_user_config()
        assert config == DEFAULT_CONFIG


# LLM-generated content at query #7
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "test_config.yaml"
    config_content = {
        "cookiecutters_dir": "~/test_dir/",
        "replay_dir": "~/test_replay/",
        "default_context": {"key": "value"},
        "abbreviations": {"custom": "https://custom.com/{0}"}
    }
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump(config_content, f)

    result = get_config(config_file)
    assert result["cookiecutters_dir"] == os.path.expanduser("~/test_dir/")
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
        f.write("- list item")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #8
#--------------------------

```python
def test_get_config(mocker, tmp_path):
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
    assert result['default_context'] == collections.OrderedDict([('key', 'value')])
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

    # Test with environment variable expansion
    env_var_file = tmp_path / 'env_var.yaml'
    env_var_content = {
        'cookiecutters_dir': '$HOME/test_cookiecutters/',
        'replay_dir': '$USER/test_replay/'
    }
    env_var_file.write_text(yaml.dump(env_var_content))
    result = get_config(env_var_file)
    assert result['cookiecutters_dir'] == os.path.expandvars('$HOME/test_cookiecutters/')
    assert result['replay_dir'] == os.path.expandvars('$USER/test_replay/')


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


# LLM-generated content at query #10
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
        f.write("cookiecutters_dir: /test/dir\nabbreviations:\n  custom: 'https://custom.com/{0}'")
        f.flush()
        config = get_user_config(config_file=f.name)
        assert config['cookiecutters_dir'] == '/test/dir'
        assert config['abbreviations']['custom'] == 'https://custom.com/{0}'
        os.unlink(f.name)

    # Test environment variable
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("replay_dir: /env/dir")
        f.flush()
        os.environ['COOKIECUTTER_CONFIG'] = f.name
        config = get_user_config()
        assert config['replay_dir'] == '/env/dir'
        del os.environ['COOKIECUTTER_CONFIG']
        os.unlink(f.name)

    # Test non-existent config file raises exception
    with pytest.raises(ConfigDoesNotExistException):
        get_user_config(config_file='/non/existent/file.yaml')

    # Test invalid YAML raises exception
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("invalid: yaml: content: [")
        f.flush()
        with pytest.raises(InvalidConfiguration):
            get_user_config(config_file=f.name)
        os.unlink(f.name)


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
        'abbreviations': {'custom': 'https://custom.com/{0}'},
    }
    config_file.write_text(yaml.dump(config_content))

    result = get_config(config_file)
    assert result['cookiecutters_dir'] == os.path.expanduser('~/test_cookiecutters/')
    assert result['replay_dir'] == os.path.expanduser('~/test_replay/')
    assert result['default_context'] == {'key': 'value'}
    assert result['abbreviations'] == {
        'gh': 'https://github.com/{0}.git',
        'gl': 'https://gitlab.com/{0}.git',
        'bb': 'https://bitbucket.org/{0}',
        'custom': 'https://custom.com/{0}',
    }

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
    non_dict_yaml_file.write_text("- list item 1\n- list item 2")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #12
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
            custom: https://custom.example.com/{0}
        """
    )

    config = get_config(config_file)
    assert config["cookiecutters_dir"] == "/custom/cookiecutters/"
    assert config["replay_dir"] == "/custom/replay/"
    assert config["default_context"] == {"key1": "value1"}
    assert config["abbreviations"]["custom"] == "https://custom.example.com/{0}"
    assert config["abbreviations"]["gh"] == "https://github.com/{0}.git"  # Check default is preserved

    # Test with environment variable expansion
    config_file.write_text(
        """
        cookiecutters_dir: $HOME/test_cookiecutters/
        replay_dir: ~/test_replay/
        """
    )
    os.environ["HOME"] = "/test_home"
    config = get_config(config_file)
    assert config["cookiecutters_dir"] == "/test_home/test_cookiecutters/"
    assert config["replay_dir"] == "/test_home/test_replay/"

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


# LLM-generated content at query #13
#--------------------------

```python
def test_get_user_config():
    # Test default config
    config = get_user_config(default_config=True)
    assert config == DEFAULT_CONFIG

    # Test custom default config
    custom_default = {'cookiecutters_dir': '/custom/path'}
    config = get_user_config(default_config=custom_default)
    expected = merge_configs(DEFAULT_CONFIG, custom_default)
    assert config == expected

    # Test custom config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump({'cookiecutters_dir': '/test/path'}, f)
        f.flush()
        config = get_user_config(config_file=f.name)
        assert config['cookiecutters_dir'] == '/test/path'
        os.unlink(f.name)

    # Test environment variable
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump({'replay_dir': '/env/path'}, f)
        f.flush()
        os.environ['COOKIECUTTER_CONFIG'] = f.name
        config = get_user_config()
        assert config['replay_dir'] == '/env/path'
        del os.environ['COOKIECUTTER_CONFIG']
        os.unlink(f.name)

    # Test user config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump({'default_context': {'key': 'value'}}, f)
        f.flush()
        with patch('cookiecutter.config.USER_CONFIG_PATH', f.name):
            config = get_user_config()
            assert config['default_context'] == {'key': 'value'}
        os.unlink(f.name)


# LLM-generated content at query #14
#--------------------------

```python
def test_get_config(mocker, tmp_path):
    # Test with a valid config file
    config_file = tmp_path / 'config.yaml'
    config_content = {
        'cookiecutters_dir': '~/test_cookiecutters/',
        'replay_dir': '~/test_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'test': 'https://test.com/{0}'}
    }
    config_file.write_text(yaml.dump(config_content))

    result = get_config(config_file)
    assert result['cookiecutters_dir'] == os.path.expanduser('~/test_cookiecutters/')
    assert result['replay_dir'] == os.path.expanduser('~/test_replay/')
    assert result['default_context'] == {'key': 'value'}
    assert result['abbreviations'] == {**BUILTIN_ABBREVIATIONS, 'test': 'https://test.com/{0}'}

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


# LLM-generated content at query #15
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
        f.write("invalid: yaml: content: [unclosed")

    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with a YAML file that is not a dict
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    with open(non_dict_yaml_file, 'w', encoding='utf-8') as f:
        f.write("- list item 1\n- list item 2")

    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #16
#--------------------------

```python
def test_get_config(mocker, tmp_path):
    # Test successful config loading
    config_content = {
        'cookiecutters_dir': '~/test_dir/',
        'replay_dir': '~/test_replay/',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://custom.com/{0}'}
    }
    config_file = tmp_path / 'config.yaml'
    config_file.write_text(yaml.dump(config_content))

    result = get_config(config_file)
    assert result['cookiecutters_dir'] == os.path.expanduser('~/test_dir/')
    assert result['replay_dir'] == os.path.expanduser('~/test_replay/')
    assert result['default_context'] == {'key': 'value'}
    assert result['abbreviations'] == {**BUILTIN_ABBREVIATIONS, 'custom': 'https://custom.com/{0}'}

    # Test non-existent config file
    non_existent_file = tmp_path / 'non_existent.yaml'
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test invalid YAML
    invalid_yaml_file = tmp_path / 'invalid.yaml'
    invalid_yaml_file.write_text('invalid: yaml: content: [unclosed')
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test non-dict YAML content
    non_dict_yaml_file = tmp_path / 'non_dict.yaml'
    non_dict_yaml_file.write_text('just a string')
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)

    # Test environment variable expansion
    env_config_content = {
        'cookiecutters_dir': '$HOME/test_dir/',
        'replay_dir': '$USER/test_replay/'
    }
    env_config_file = tmp_path / 'env_config.yaml'
    env_config_file.write_text(yaml.dump(env_config_content))

    result = get_config(env_config_file)
    assert result['cookiecutters_dir'] == os.path.expandvars('$HOME/test_dir/')
    assert result['replay_dir'] == os.path.expandvars('$USER/test_replay/')


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


# LLM-generated content at query #19
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


# LLM-generated content at query #20
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with valid config file
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
    assert result['default_context'] == config_content['default_context']
    assert result['abbreviations'] == {**BUILTIN_ABBREVIATIONS, **config_content['abbreviations']}

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
        f.write("- list item 1\n- list item 2")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #21
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

    # Test fallback to default config when no config exists
    if os.path.exists(USER_CONFIG_PATH):
        os.unlink(USER_CONFIG_PATH)
    if 'COOKIECUTTER_CONFIG' in os.environ:
        del os.environ['COOKIECUTTER_CONFIG']
    config = get_user_config()
    assert config == DEFAULT_CONFIG


# LLM-generated content at query #22
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
    config_path = 'test_config.yaml'
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(valid_config, f)

    result = get_config(config_path)
    assert result['cookiecutters_dir'] == os.path.expanduser('~/.test_cookiecutters/')
    assert result['replay_dir'] == os.path.expanduser('~/.test_replay/')
    assert result['default_context'] == {'key': 'value'}
    assert result['abbreviations'] == {**BUILTIN_ABBREVIATIONS, 'test': 'https://test.com/{0}'}

    # Test with a non-existent config file
    with pytest.raises(ConfigDoesNotExistException):
        get_config('non_existent_config.yaml')

    # Test with an invalid YAML file
    with open('invalid_config.yaml', 'w', encoding='utf-8') as f:
        f.write('invalid: yaml: content: [')
    with pytest.raises(InvalidConfiguration):
        get_config('invalid_config.yaml')

    # Test with a YAML file that is not a dict
    with open('non_dict_config.yaml', 'w', encoding='utf-8') as f:
        f.write('- list item')
    with pytest.raises(InvalidConfiguration):
        get_config('non_dict_config.yaml')


# LLM-generated content at query #23
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


# LLM-generated content at query #24
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
    with open(config_file, 'w', encoding='utf-8') as f:
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


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
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
    assert result['default_context'] == collections.OrderedDict([('key', 'value')])
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
        f.write("- list_item1\n- list_item2")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #2
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
        get_user_config(config_file='/non/existent/file.yaml')


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
        f.write("invalid: yaml: content: [unclosed")
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
def test_get_user_config():
    # Test default config
    config = get_user_config(default_config=True)
    assert config == DEFAULT_CONFIG

    # Test custom config dict
    custom_config = {'cookiecutters_dir': '/custom/path'}
    config = get_user_config(default_config=custom_config)
    expected = merge_configs(DEFAULT_CONFIG, custom_config)
    assert config == expected

    # Test loading from custom config file
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write('cookiecutters_dir: /test/path')
        f.flush()
        config = get_user_config(config_file=f.name)
        assert config['cookiecutters_dir'] == '/test/path'
        os.unlink(f.name)

    # Test loading from environment variable
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write('cookiecutters_dir: /env/path')
        f.flush()
        os.environ['COOKIECUTTER_CONFIG'] = f.name
        config = get_user_config()
        assert config['cookiecutters_dir'] == '/env/path'
        del os.environ['COOKIECUTTER_CONFIG']
        os.unlink(f.name)

    # Test loading from default user config path
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write('cookiecutters_dir: /user/path')
        f.flush()
        os.rename(f.name, USER_CONFIG_PATH)
        config = get_user_config()
        assert config['cookiecutters_dir'] == '/user/path'
        os.unlink(USER_CONFIG_PATH)


# LLM-generated content at query #5
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

    config = get_config(config_file)
    assert config['cookiecutters_dir'] == os.path.expanduser('~/test_cookiecutters/')
    assert config['replay_dir'] == os.path.expanduser('~/test_replay/')
    assert config['default_context'] == {'key': 'value'}
    assert config['abbreviations'] == {**BUILTIN_ABBREVIATIONS, **{'test': 'https://test.com/{0}'}}

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


# LLM-generated content at query #6
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
        f.write("invalid: yaml: content: [unclosed")

    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with a YAML file that is not a dict
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    with open(non_dict_yaml_file, 'w', encoding='utf-8') as f:
        f.write("- list item")

    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


# LLM-generated content at query #7
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


# LLM-generated content at query #8
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


# LLM-generated content at query #10
#--------------------------

```python
def test_get_user_config():
    # Test default config
    config = get_user_config(default_config=True)
    assert config == DEFAULT_CONFIG

    # Test custom default config
    custom_default = {'cookiecutters_dir': '/custom/path'}
    config = get_user_config(default_config=custom_default)
    expected = merge_configs(DEFAULT_CONFIG, custom_default)
    assert config == expected

    # Test loading from custom config file
    with pytest.raises(ConfigDoesNotExistException):
        get_user_config(config_file='/nonexistent/path')

    # Test loading from environment variable
    with pytest.raises(ConfigDoesNotExistException):
        with patch.dict(os.environ, {'COOKIECUTTER_CONFIG': '/nonexistent/path'}):
            get_user_config()

    # Test loading from default user config path
    with patch('os.path.exists', return_value=True):
        with patch('cookiecutter.config.get_config', return_value={'test': 'value'}):
            config = get_user_config()
            assert config == {'test': 'value'}

    # Test fallback to default config when no user config exists
    with patch('os.path.exists', return_value=False):
        config = get_user_config()
        assert config == DEFAULT_CONFIG


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


# LLM-generated content at query #12
#--------------------------

```python
def test_get_config(tmp_path):
    # Test with a valid config file
    config_file = tmp_path / "cookiecutterrc"
    config_file.write_text("""
        cookiecutters_dir: /custom/cookiecutters/
        replay_dir: /custom/replay/
        default_context:
            key1: value1
        abbreviations:
            custom: https://custom.com/{0}
    """)

    config = get_config(config_file)
    assert config['cookiecutters_dir'] == '/custom/cookiecutters/'
    assert config['replay_dir'] == '/custom/replay/'
    assert config['default_context'] == {'key1': 'value1'}
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
    non_existent_file = tmp_path / "non_existent"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test with invalid YAML
    invalid_yaml_file = tmp_path / "invalid_yaml"
    invalid_yaml_file.write_text("invalid: yaml: content: [unclosed")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with non-dict YAML content
    non_dict_yaml_file = tmp_path / "non_dict_yaml"
    non_dict_yaml_file.write_text("- list item")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


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


# LLM-generated content at query #14
#--------------------------

```python
def test_get_config():
    # Test with a valid config file
    valid_config_file = "valid_config.yaml"
    with open(valid_config_file, "w", encoding="utf-8") as f:
        f.write("cookiecutters_dir: /custom/cookiecutters/\nreplay_dir: /custom/replay/\n")

    config = get_config(valid_config_file)
    assert config["cookiecutters_dir"] == "/custom/cookiecutters/"
    assert config["replay_dir"] == "/custom/replay/"
    assert config["default_context"] == collections.OrderedDict([])
    assert config["abbreviations"] == BUILTIN_ABBREVIATIONS

    # Test with a non-existent config file
    with pytest.raises(ConfigDoesNotExistException):
        get_config("non_existent_config.yaml")

    # Test with an invalid YAML file
    invalid_yaml_file = "invalid_yaml.yaml"
    with open(invalid_yaml_file, "w", encoding="utf-8") as f:
        f.write("invalid: yaml: content: [")

    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with a YAML file that has a non-dict top-level element
    non_dict_yaml_file = "non_dict_yaml.yaml"
    with open(non_dict_yaml_file, "w", encoding="utf-8") as f:
        f.write("- list item 1\n- list item 2")

    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)

    # Test with environment variable expansion
    env_config_file = "env_config.yaml"
    with open(env_config_file, "w", encoding="utf-8") as f:
        f.write("cookiecutters_dir: $HOME/test/cookiecutters/\nreplay_dir: ~/test/replay/\n")

    config = get_config(env_config_file)
    assert config["cookiecutters_dir"] == os.path.expandvars("$HOME/test/cookiecutters/")
    assert config["replay_dir"] == os.path.expanduser("~/test/replay/")


# LLM-generated content at query #15
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
    assert config['default_context'] == collections.OrderedDict([('key', 'value')])
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


# LLM-generated content at query #16
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
        f.write("cookiecutters_dir: /custom/dir\n")
        f.flush()
        config = get_user_config(config_file=f.name)
        assert config['cookiecutters_dir'] == '/custom/dir'
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

    # Test user config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("cookiecutters_dir: /user/dir\n")
        f.flush()
        os.rename(f.name, USER_CONFIG_PATH)
        config = get_user_config()
        assert config['cookiecutters_dir'] == '/user/dir'
        os.unlink(USER_CONFIG_PATH)


# LLM-generated content at query #17
#--------------------------

```python
def test_get_config():
    # Test with a valid config file
    valid_config_file = "tests/data/valid_config.yaml"
    config = get_config(valid_config_file)
    assert config["cookiecutters_dir"] == os.path.expanduser("~/.cookiecutters/")
    assert config["replay_dir"] == os.path.expanduser("~/.cookiecutter_replay/")
    assert config["default_context"] == collections.OrderedDict([])
    assert config["abbreviations"] == BUILTIN_ABBREVIATIONS

    # Test with a non-existent config file
    non_existent_config_file = "tests/data/non_existent_config.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_config_file)

    # Test with an invalid YAML file
    invalid_yaml_file = "tests/data/invalid_yaml.yaml"
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with a YAML file that is not a dict
    non_dict_yaml_file = "tests/data/non_dict_yaml.yaml"
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)

    # Test with a config file that has environment variables
    config_with_env_vars = "tests/data/config_with_env_vars.yaml"
    config = get_config(config_with_env_vars)
    assert config["cookiecutters_dir"] == os.path.expanduser("~/.cookiecutters/")
    assert config["replay_dir"] == os.path.expanduser("~/.cookiecutter_replay/")


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
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config_content, f)

    config = get_config(config_file)
    assert config['cookiecutters_dir'] == os.path.expanduser('~/test_cookiecutters/')
    assert config['replay_dir'] == os.path.expanduser('~/test_replay/')
    assert config['default_context'] == {'key': 'value'}
    assert config['abbreviations'] == {**BUILTIN_ABBREVIATIONS, **{'custom': 'https://custom.com/{0}'}}

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


# LLM-generated content at query #19
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
    invalid_yaml_file.write_text("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test with a YAML file that is not a dict
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    non_dict_yaml_file.write_text("- list item")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)


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
        f.write("- list item 1\n- list item 2")
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


# LLM-generated content at query #22
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


