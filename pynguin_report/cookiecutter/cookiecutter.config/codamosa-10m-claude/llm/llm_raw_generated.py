####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_get_config(tmp_path, monkeypatch):
    """Test get_config function with various scenarios."""
    
    # Test 1: Config file does not exist
    with pytest.raises(ConfigDoesNotExistException):
        get_config('/nonexistent/path/config.yaml')
    
    # Test 2: Valid YAML config file
    config_file = tmp_path / "valid_config.yaml"
    config_file.write_text(
        "cookiecutters_dir: /custom/path\n"
        "replay_dir: /replay/path\n"
        "default_context:\n"
        "  author_name: John Doe\n"
        "abbreviations:\n"
        "  custom: https://example.com/{0}.git\n"
    )
    config = get_config(config_file)
    
    assert config['cookiecutters_dir'] == '/custom/path'
    assert config['replay_dir'] == '/replay/path'
    assert config['default_context']['author_name'] == 'John Doe'
    assert config['abbreviations']['custom'] == 'https://example.com/{0}.git'
    assert config['abbreviations']['gh'] == 'https://github.com/{0}.git'  # Merged defaults
    
    # Test 3: Empty YAML file
    empty_config_file = tmp_path / "empty_config.yaml"
    empty_config_file.write_text("")
    config = get_config(empty_config_file)
    
    assert config == DEFAULT_CONFIG
    
    # Test 4: Path expansion with environment variables
    monkeypatch.setenv('TEST_DIR', '/test/dir')
    config_with_env = tmp_path / "config_with_env.yaml"
    config_with_env.write_text("cookiecutters_dir: $TEST_DIR/cookies\n")
    config = get_config(config_with_env)
    
    assert '/test/dir/cookies' in config['cookiecutters_dir']
    
    # Test 5: Path expansion with home directory
    config_with_home = tmp_path / "config_with_home.yaml"
    config_with_home.write_text("replay_dir: ~/my_replay\n")
    config = get_config(config_with_home)
    
    assert '~' not in config['replay_dir']
    assert config['replay_dir'].startswith(os.path.expanduser('~'))
    
    # Test 6: Invalid YAML syntax
    invalid_yaml_file = tmp_path / "invalid.yaml"
    invalid_yaml_file.write_text("invalid: yaml: content: [")
    
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)
    
    # Test 7: Top-level element is not a dict
    non_dict_yaml = tmp_path / "non_dict.yaml"
    non_dict_yaml.write_text("- item1\n- item2\n")
    
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml)
    
    # Test 8: Partial config merges with defaults
    partial_config_file = tmp_path / "partial_config.yaml"
    partial_config_file.write_text("default_context:\n  key1: value1\n")
    config = get_config(partial_config_file)
    
    assert config['default_context']['key1'] == 'value1'
    assert 'cookiecutters_dir' in config
    assert 'replay_dir' in config
    assert 'abbreviations' in config
    
    # Test 9: Config file path as string and Path object
    config_path_str = str(config_file)
    config1 = get_config(config_path_str)
    config2 = get_config(config_file)
    
    assert config1['cookiecutters_dir'] == config2['cookiecutters_dir']


# LLM-generated content at query #2
#--------------------------

```python
def test_get_config(tmp_path, monkeypatch):
    """Test get_config function with various scenarios."""
    # Test 1: Valid YAML config file
    config_file = tmp_path / "config.yaml"
    config_content = {
        'cookiecutters_dir': '/custom/cookiecutters',
        'replay_dir': '/custom/replay',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://example.com/{0}'}
    }
    config_file.write_text(yaml.dump(config_content))
    
    result = get_config(config_file)
    assert result['cookiecutters_dir'] == '/custom/cookiecutters'
    assert result['replay_dir'] == '/custom/replay'
    assert result['default_context'] == {'key': 'value'}
    assert result['abbreviations']['custom'] == 'https://example.com/{0}'
    # Built-in abbreviations should be preserved
    assert result['abbreviations']['gh'] == 'https://github.com/{0}.git'

    # Test 2: Config file with environment variables
    config_file2 = tmp_path / "config2.yaml"
    monkeypatch.setenv('TEST_DIR', '/test/path')
    config_content2 = {
        'cookiecutters_dir': '$TEST_DIR/cookiecutters',
        'replay_dir': '~/replay'
    }
    config_file2.write_text(yaml.dump(config_content2))
    
    result2 = get_config(config_file2)
    assert result2['cookiecutters_dir'] == '/test/path/cookiecutters'
    assert '/replay' in result2['replay_dir']  # Home expanded

    # Test 3: Non-existent config file
    non_existent = tmp_path / "nonexistent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent)

    # Test 4: Invalid YAML syntax
    invalid_yaml = tmp_path / "invalid.yaml"
    invalid_yaml.write_text("{ invalid yaml content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml)

    # Test 5: Non-dict YAML content
    non_dict_yaml = tmp_path / "non_dict.yaml"
    non_dict_yaml.write_text("- item1\n- item2")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml)

    # Test 6: Empty YAML file
    empty_yaml = tmp_path / "empty.yaml"
    empty_yaml.write_text("")
    result3 = get_config(empty_yaml)
    assert result3['cookiecutters_dir'] == DEFAULT_CONFIG['cookiecutters_dir']
    assert result3['replay_dir'] == DEFAULT_CONFIG['replay_dir']

    # Test 7: Partial config (only some keys)
    partial_yaml = tmp_path / "partial.yaml"
    partial_yaml.write_text("default_context:\n  author: Test Author")
    result4 = get_config(partial_yaml)
    assert result4['default_context']['author'] == 'Test Author'
    assert result4['cookiecutters_dir'] == DEFAULT_CONFIG['cookiecutters_dir']


# LLM-generated content at query #3
#--------------------------

```python
def test_get_config(tmp_path):
    """Test get_config function with various scenarios."""
    import pytest
    
    # Test 1: Config file does not exist
    with pytest.raises(ConfigDoesNotExistException):
        get_config('/nonexistent/path/config.yaml')
    
    # Test 2: Valid YAML config file
    config_file = tmp_path / 'config.yaml'
    config_file.write_text(
        'cookiecutters_dir: /tmp/cookies\n'
        'replay_dir: /tmp/replay\n'
        'default_context:\n'
        '  author: Test Author\n'
        'abbreviations:\n'
        '  custom: https://example.com/{0}.git\n'
    )
    result = get_config(config_file)
    assert result['cookiecutters_dir'] == '/tmp/cookies'
    assert result['replay_dir'] == '/tmp/replay'
    assert result['default_context']['author'] == 'Test Author'
    assert result['abbreviations']['custom'] == 'https://example.com/{0}.git'
    assert result['abbreviations']['gh'] == 'https://github.com/{0}.git'
    
    # Test 3: Empty YAML file
    empty_config = tmp_path / 'empty.yaml'
    empty_config.write_text('')
    result = get_config(empty_config)
    assert result == DEFAULT_CONFIG
    
    # Test 4: Invalid YAML syntax
    invalid_config = tmp_path / 'invalid.yaml'
    invalid_config.write_text('invalid: yaml: content:')
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_config)
    
    # Test 5: Non-dict top-level element
    non_dict_config = tmp_path / 'nondict.yaml'
    non_dict_config.write_text('- item1\n- item2\n')
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_config)
    
    # Test 6: Path expansion with environment variables
    config_with_env = tmp_path / 'env_config.yaml'
    config_with_env.write_text('cookiecutters_dir: $HOME/.cookiecutters\n')
    result = get_config(config_with_env)
    assert '$HOME' not in result['cookiecutters_dir']
    
    # Test 7: Path expansion with tilde
    config_with_tilde = tmp_path / 'tilde_config.yaml'
    config_with_tilde.write_text('replay_dir: ~/custom_replay\n')
    result = get_config(config_with_tilde)
    assert '~' not in result['replay_dir']
    assert result['replay_dir'].startswith(os.path.expanduser('~'))
    
    # Test 8: Merging with defaults
    minimal_config = tmp_path / 'minimal.yaml'
    minimal_config.write_text('cookiecutters_dir: /custom/path\n')
    result = get_config(minimal_config)
    assert result['cookiecutters_dir'] == '/custom/path'
    assert 'replay_dir' in result
    assert 'default_context' in result


# LLM-generated content at query #4
#--------------------------

```python
def test_get_config(tmp_path, monkeypatch):
    """Test get_config function with valid and invalid configurations."""
    # Test with valid YAML config file
    config_file = tmp_path / "config.yaml"
    config_content = """
cookiecutters_dir: ~/custom_cookiecutters/
replay_dir: ~/custom_replay/
default_context:
    author_name: John Doe
abbreviations:
    gh: https://github.com/{0}.git
"""
    config_file.write_text(config_content)
    
    result = get_config(config_file)
    
    assert isinstance(result, dict)
    assert 'cookiecutters_dir' in result
    assert 'replay_dir' in result
    assert result['default_context']['author_name'] == 'John Doe'
    assert result['abbreviations']['gh'] == 'https://github.com/{0}.git'
    assert os.path.expanduser('~') in result['cookiecutters_dir']
    assert os.path.expanduser('~') in result['replay_dir']


def test_get_config_nonexistent_file():
    """Test get_config raises exception for nonexistent file."""
    with pytest.raises(ConfigDoesNotExistException):
        get_config('/nonexistent/path/config.yaml')


def test_get_config_invalid_yaml(tmp_path):
    """Test get_config raises exception for invalid YAML."""
    config_file = tmp_path / "invalid.yaml"
    config_file.write_text("invalid: yaml: content: [")
    
    with pytest.raises(InvalidConfiguration):
        get_config(config_file)


def test_get_config_non_dict_yaml(tmp_path):
    """Test get_config raises exception when YAML is not a dict."""
    config_file = tmp_path / "list.yaml"
    config_file.write_text("- item1\n- item2\n")
    
    with pytest.raises(InvalidConfiguration):
        get_config(config_file)


def test_get_config_empty_file(tmp_path):
    """Test get_config with empty YAML file."""
    config_file = tmp_path / "empty.yaml"
    config_file.write_text("")
    
    result = get_config(config_file)
    
    assert result == DEFAULT_CONFIG


def test_get_config_merges_with_defaults(tmp_path):
    """Test get_config merges user config with defaults."""
    config_file = tmp_path / "partial.yaml"
    config_file.write_text("default_context:\n    custom_key: custom_value\n")
    
    result = get_config(config_file)
    
    assert 'cookiecutters_dir' in result
    assert 'replay_dir' in result
    assert result['default_context']['custom_key'] == 'custom_value'
    assert 'gh' in result['abbreviations']


def test_get_config_expands_environment_variables(tmp_path, monkeypatch):
    """Test get_config expands environment variables in paths."""
    monkeypatch.setenv('CUSTOM_DIR', '/custom/path')
    config_file = tmp_path / "envvar.yaml"
    config_file.write_text("cookiecutters_dir: $CUSTOM_DIR/cookiecutters\n")
    
    result = get_config(config_file)
    
    assert '/custom/path/cookiecutters' in result['cookiecutters_dir']


def test_get_config_overrides_abbreviations(tmp_path):
    """Test get_config can override builtin abbreviations."""
    config_file = tmp_path / "abbrev.yaml"
    config_file.write_text("abbreviations:\n    gh: https://custom.github.com/{0}.git\n")
    
    result = get_config(config_file)
    
    assert result['abbreviations']['gh'] == 'https://custom.github.com/{0}.git'
    assert result['abbreviations']['gl'] == 'https://gitlab.com/{0}.git'


# LLM-generated content at query #5
#--------------------------

```python
def test_get_config(tmp_path, monkeypatch):
    """Test get_config function with various scenarios."""
    
    # Test 1: Non-existent config file raises ConfigDoesNotExistException
    with pytest.raises(ConfigDoesNotExistException):
        get_config('/nonexistent/path/config.yaml')
    
    # Test 2: Valid YAML config file is loaded and merged with defaults
    config_file = tmp_path / 'config.yaml'
    config_file.write_text(
        'cookiecutters_dir: /custom/path\n'
        'default_context:\n'
        '  project_name: my_project\n'
    )
    result = get_config(config_file)
    assert result['cookiecutters_dir'] == '/custom/path'
    assert result['default_context']['project_name'] == 'my_project'
    assert 'replay_dir' in result  # From defaults
    assert result['abbreviations'] == BUILTIN_ABBREVIATIONS  # From defaults
    
    # Test 3: Empty YAML file uses all defaults
    empty_config = tmp_path / 'empty_config.yaml'
    empty_config.write_text('')
    result = get_config(empty_config)
    assert result == DEFAULT_CONFIG
    
    # Test 4: Path expansion with environment variables
    config_file = tmp_path / 'config_with_env.yaml'
    monkeypatch.setenv('TEST_DIR', '/test/directory')
    config_file.write_text('cookiecutters_dir: $TEST_DIR/cookies\n')
    result = get_config(config_file)
    assert result['cookiecutters_dir'] == '/test/directory/cookies'
    
    # Test 5: Path expansion with home directory
    config_file = tmp_path / 'config_with_home.yaml'
    config_file.write_text('cookiecutters_dir: ~/my_cookies\n')
    result = get_config(config_file)
    assert '~' not in result['cookiecutters_dir']
    assert result['cookiecutters_dir'].startswith(os.path.expanduser('~'))
    
    # Test 6: Invalid YAML raises InvalidConfiguration
    invalid_yaml = tmp_path / 'invalid.yaml'
    invalid_yaml.write_text('invalid: yaml: content: [')
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml)
    
    # Test 7: Non-dict top-level element raises InvalidConfiguration
    non_dict_yaml = tmp_path / 'non_dict.yaml'
    non_dict_yaml.write_text('- item1\n- item2\n')
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml)
    
    # Test 8: Nested dict values are merged with defaults
    config_file = tmp_path / 'nested_config.yaml'
    config_file.write_text(
        'abbreviations:\n'
        '  custom: https://custom.com/{0}\n'
    )
    result = get_config(config_file)
    assert 'gh' in result['abbreviations']  # From defaults
    assert result['abbreviations']['custom'] == 'https://custom.com/{0}'
    
    # Test 9: replay_dir is expanded
    config_file = tmp_path / 'replay_config.yaml'
    config_file.write_text('replay_dir: ~/my_replay\n')
    result = get_config(config_file)
    assert '~' not in result['replay_dir']
    assert result['replay_dir'].startswith(os.path.expanduser('~'))


# LLM-generated content at query #6
#--------------------------

```python
def test_get_config(tmp_path, monkeypatch):
    """Test get_config function with various scenarios."""
    
    # Test 1: Config file does not exist
    with pytest.raises(ConfigDoesNotExistException):
        get_config('/nonexistent/path/config.yaml')
    
    # Test 2: Valid config file with basic content
    config_file = tmp_path / "config.yaml"
    config_file.write_text("cookiecutters_dir: /tmp/cookiecutters\nreplay_dir: /tmp/replay")
    
    result = get_config(config_file)
    assert result['cookiecutters_dir'] == '/tmp/cookiecutters'
    assert result['replay_dir'] == '/tmp/replay'
    assert 'abbreviations' in result
    assert result['abbreviations'] == BUILTIN_ABBREVIATIONS
    
    # Test 3: Config file with environment variable expansion
    config_file = tmp_path / "config_with_env.yaml"
    config_file.write_text("cookiecutters_dir: $HOME/.cookiecutters\nreplay_dir: ~/replay")
    
    result = get_config(config_file)
    assert '$HOME' not in result['cookiecutters_dir']
    assert '~' not in result['replay_dir']
    
    # Test 4: Empty config file
    empty_config = tmp_path / "empty.yaml"
    empty_config.write_text("")
    
    result = get_config(empty_config)
    assert result['cookiecutters_dir'] == os.path.expanduser('~/.cookiecutters/')
    assert result['replay_dir'] == os.path.expanduser('~/.cookiecutter_replay/')
    
    # Test 5: Config with merged abbreviations
    config_file = tmp_path / "config_abbrev.yaml"
    config_file.write_text("abbreviations:\n  custom: https://example.com/{0}.git")
    
    result = get_config(config_file)
    assert 'gh' in result['abbreviations']
    assert 'custom' in result['abbreviations']
    assert result['abbreviations']['custom'] == 'https://example.com/{0}.git'
    
    # Test 6: Invalid YAML syntax
    invalid_yaml = tmp_path / "invalid.yaml"
    invalid_yaml.write_text("invalid: yaml: content: [")
    
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml)
    
    # Test 7: YAML file with non-dict top-level element
    non_dict_yaml = tmp_path / "non_dict.yaml"
    non_dict_yaml.write_text("- item1\n- item2")
    
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml)
    
    # Test 8: Config with default_context
    config_file = tmp_path / "config_context.yaml"
    config_file.write_text("default_context:\n  author_name: John Doe\n  project_name: My Project")
    
    result = get_config(config_file)
    assert result['default_context']['author_name'] == 'John Doe'
    assert result['default_context']['project_name'] == 'My Project'
    
    # Test 9: Path expansion for user home
    config_file = tmp_path / "config_expand.yaml"
    config_file.write_text("cookiecutters_dir: ~/custom_cookiecutters")
    
    result = get_config(config_file)
    assert '~' not in result['cookiecutters_dir']
    assert result['cookiecutters_dir'].startswith(os.path.expanduser('~'))


# LLM-generated content at query #7
#--------------------------

```python
def test_get_config(tmp_path, monkeypatch):
    """Test get_config function with various scenarios."""
    # Test 1: Config file does not exist
    non_existent_path = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_path)

    # Test 2: Valid YAML config file
    valid_config_path = tmp_path / "valid_config.yaml"
    valid_config_content = """
cookiecutters_dir: ~/.cookiecutters/
replay_dir: ~/.cookiecutter_replay/
default_context:
  author_name: John Doe
abbreviations:
  gh: https://github.com/{0}.git
"""
    valid_config_path.write_text(valid_config_content)
    config = get_config(valid_config_path)
    
    assert isinstance(config, dict)
    assert 'cookiecutters_dir' in config
    assert 'replay_dir' in config
    assert config['default_context']['author_name'] == 'John Doe'
    assert config['abbreviations']['gh'] == 'https://github.com/{0}.git'

    # Test 3: Invalid YAML syntax
    invalid_yaml_path = tmp_path / "invalid.yaml"
    invalid_yaml_path.write_text("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_path)

    # Test 4: YAML file with non-dict top-level element
    non_dict_yaml_path = tmp_path / "non_dict.yaml"
    non_dict_yaml_path.write_text("- item1\n- item2\n")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_path)

    # Test 5: Empty YAML file
    empty_yaml_path = tmp_path / "empty.yaml"
    empty_yaml_path.write_text("")
    config = get_config(empty_yaml_path)
    assert config == DEFAULT_CONFIG

    # Test 6: Path expansion with environment variables
    env_var_config_path = tmp_path / "env_var_config.yaml"
    monkeypatch.setenv('TEST_HOME', str(tmp_path))
    env_var_config_content = """
cookiecutters_dir: $TEST_HOME/cookiecutters
replay_dir: ~/replay
"""
    env_var_config_path.write_text(env_var_config_content)
    config = get_config(env_var_config_path)
    
    assert str(tmp_path) in config['cookiecutters_dir']
    assert config['replay_dir'].startswith(os.path.expanduser('~'))

    # Test 7: Partial config merges with defaults
    partial_config_path = tmp_path / "partial_config.yaml"
    partial_config_path.write_text("default_context:\n  custom_key: custom_value\n")
    config = get_config(partial_config_path)
    
    assert config['default_context']['custom_key'] == 'custom_value'
    assert 'cookiecutters_dir' in config
    assert 'replay_dir' in config

    # Test 8: Config with nested abbreviations
    nested_abbrev_path = tmp_path / "nested_abbrev.yaml"
    nested_abbrev_content = """
abbreviations:
  gh: https://custom-github.com/{0}.git
  custom: https://custom.com/{0}
"""
    nested_abbrev_path.write_text(nested_abbrev_content)
    config = get_config(nested_abbrev_path)
    
    assert config['abbreviations']['gh'] == 'https://custom-github.com/{0}.git'
    assert config['abbreviations']['custom'] == 'https://custom.com/{0}'
    assert config['abbreviations']['bb'] == 'https://bitbucket.org/{0}'


# LLM-generated content at query #8
#--------------------------

```python
def test_get_config(tmp_path, monkeypatch):
    """Test get_config function with various scenarios."""
    # Test 1: Config file does not exist
    non_existent_path = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_path)

    # Test 2: Valid config file with basic content
    config_file = tmp_path / "config.yaml"
    config_content = {
        'cookiecutters_dir': '/tmp/cookiecutters',
        'replay_dir': '/tmp/replay',
        'default_context': {'key': 'value'},
    }
    config_file.write_text(yaml.dump(config_content))
    result = get_config(config_file)
    assert result['default_context'] == {'key': 'value'}
    assert '/tmp/cookiecutters' in result['cookiecutters_dir']
    assert '/tmp/replay' in result['replay_dir']

    # Test 3: Config file with environment variables in paths
    config_file_with_env = tmp_path / "config_env.yaml"
    monkeypatch.setenv('TEST_DIR', '/test/path')
    config_content_env = {
        'cookiecutters_dir': '$TEST_DIR/cookiecutters',
        'replay_dir': '~/replay',
    }
    config_file_with_env.write_text(yaml.dump(config_content_env))
    result = get_config(config_file_with_env)
    assert '/test/path/cookiecutters' in result['cookiecutters_dir']
    assert os.path.expanduser('~') in result['replay_dir']

    # Test 4: Config file with abbreviations
    config_file_abbrev = tmp_path / "config_abbrev.yaml"
    config_content_abbrev = {
        'abbreviations': {
            'custom': 'https://custom.com/{0}.git',
        }
    }
    config_file_abbrev.write_text(yaml.dump(config_content_abbrev))
    result = get_config(config_file_abbrev)
    assert 'gh' in result['abbreviations']
    assert result['abbreviations']['custom'] == 'https://custom.com/{0}.git'

    # Test 5: Invalid YAML syntax
    config_file_invalid = tmp_path / "config_invalid.yaml"
    config_file_invalid.write_text("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(config_file_invalid)

    # Test 6: Non-dict top-level element
    config_file_non_dict = tmp_path / "config_non_dict.yaml"
    config_file_non_dict.write_text("- item1\n- item2\n")
    with pytest.raises(InvalidConfiguration):
        get_config(config_file_non_dict)

    # Test 7: Empty YAML file
    config_file_empty = tmp_path / "config_empty.yaml"
    config_file_empty.write_text("")
    result = get_config(config_file_empty)
    assert result['cookiecutters_dir'] == os.path.expanduser('~/.cookiecutters/')
    assert result['replay_dir'] == os.path.expanduser('~/.cookiecutter_replay/')
    assert 'gh' in result['abbreviations']

    # Test 8: Partial config file (should merge with defaults)
    config_file_partial = tmp_path / "config_partial.yaml"
    config_content_partial = {
        'default_context': {'author': 'John Doe'},
    }
    config_file_partial.write_text(yaml.dump(config_content_partial))
    result = get_config(config_file_partial)
    assert result['default_context'] == {'author': 'John Doe'}
    assert result['cookiecutters_dir'] == os.path.expanduser('~/.cookiecutters/')
    assert 'gh' in result['abbreviations']


# LLM-generated content at query #9
#--------------------------

```python
def test_get_config(tmp_path, monkeypatch):
    """Test get_config function with various scenarios."""
    
    # Test 1: Valid YAML config file
    config_file = tmp_path / "config.yaml"
    config_content = """
cookiecutters_dir: /custom/cookiecutters
replay_dir: /custom/replay
default_context:
    author_name: Test Author
abbreviations:
    custom: https://custom.com/{0}.git
"""
    config_file.write_text(config_content)
    
    result = get_config(config_file)
    
    assert result['cookiecutters_dir'] == '/custom/cookiecutters'
    assert result['replay_dir'] == '/custom/replay'
    assert result['default_context']['author_name'] == 'Test Author'
    assert result['abbreviations']['custom'] == 'https://custom.com/{0}.git'
    assert result['abbreviations']['gh'] == 'https://github.com/{0}.git'  # Builtin preserved
    
    # Test 2: Non-existent config file raises exception
    non_existent = tmp_path / "non_existent.yaml"
    
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent)
    
    # Test 3: Invalid YAML raises exception
    invalid_yaml_file = tmp_path / "invalid.yaml"
    invalid_yaml_file.write_text("invalid: yaml: content: [")
    
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)
    
    # Test 4: Non-dict top-level element raises exception
    non_dict_file = tmp_path / "non_dict.yaml"
    non_dict_file.write_text("- item1\n- item2")
    
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_file)
    
    # Test 5: Empty YAML file uses defaults
    empty_file = tmp_path / "empty.yaml"
    empty_file.write_text("")
    
    result = get_config(empty_file)
    
    assert 'cookiecutters_dir' in result
    assert 'replay_dir' in result
    assert 'default_context' in result
    assert 'abbreviations' in result
    
    # Test 6: Path expansion with environment variables
    config_with_env = tmp_path / "config_env.yaml"
    config_with_env.write_text("cookiecutters_dir: $HOME/.cookiecutters")
    
    result = get_config(config_with_env)
    
    assert '$HOME' not in result['cookiecutters_dir']
    assert result['cookiecutters_dir'].startswith(os.path.expanduser('~'))
    
    # Test 7: Path expansion with tilde
    config_with_tilde = tmp_path / "config_tilde.yaml"
    config_with_tilde.write_text("replay_dir: ~/.cookiecutter_replay_custom")
    
    result = get_config(config_with_tilde)
    
    assert '~' not in result['replay_dir']
    assert result['replay_dir'].startswith(os.path.expanduser('~'))


# LLM-generated content at query #10
#--------------------------

```python
def test_get_config(tmp_path, monkeypatch):
    """Test get_config function with valid and invalid configurations."""
    # Test 1: Valid YAML config file
    config_file = tmp_path / "valid_config.yaml"
    config_content = """
cookiecutters_dir: ~/my_cookiecutters/
replay_dir: ~/my_replay/
default_context:
  author_name: John Doe
abbreviations:
  custom: https://example.com/{0}.git
"""
    config_file.write_text(config_content, encoding='utf-8')
    
    result = get_config(config_file)
    
    assert 'cookiecutters_dir' in result
    assert 'replay_dir' in result
    assert result['default_context']['author_name'] == 'John Doe'
    assert result['abbreviations']['custom'] == 'https://example.com/{0}.git'
    assert result['abbreviations']['gh'] == 'https://github.com/{0}.git'  # Builtin preserved
    assert os.path.expanduser('~') in result['cookiecutters_dir']
    assert os.path.expanduser('~') in result['replay_dir']

    # Test 2: Non-existent config file
    non_existent = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent)

    # Test 3: Invalid YAML syntax
    invalid_yaml_file = tmp_path / "invalid.yaml"
    invalid_yaml_file.write_text("invalid: yaml: content: [", encoding='utf-8')
    
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test 4: YAML file with non-dict top-level element
    invalid_structure_file = tmp_path / "invalid_structure.yaml"
    invalid_structure_file.write_text("- item1\n- item2\n", encoding='utf-8')
    
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_structure_file)

    # Test 5: Empty YAML file
    empty_file = tmp_path / "empty.yaml"
    empty_file.write_text("", encoding='utf-8')
    
    result = get_config(empty_file)
    assert result == DEFAULT_CONFIG

    # Test 6: Config with environment variables in paths
    env_config_file = tmp_path / "env_config.yaml"
    env_config_file.write_text("cookiecutters_dir: $HOME/.custom_cookiecutters/", encoding='utf-8')
    
    result = get_config(env_config_file)
    assert os.path.expanduser('~') in result['cookiecutters_dir']
    assert '$HOME' not in result['cookiecutters_dir']

    # Test 7: Config with only some fields specified
    partial_config_file = tmp_path / "partial.yaml"
    partial_config_file.write_text("cookiecutters_dir: /custom/path/", encoding='utf-8')
    
    result = get_config(partial_config_file)
    assert result['cookiecutters_dir'] == '/custom/path/'
    assert 'replay_dir' in result
    assert 'default_context' in result
    assert 'abbreviations' in result


# LLM-generated content at query #11
#--------------------------

```python
def test_get_config(tmp_path, monkeypatch):
    """Test get_config function with valid YAML config file."""
    # Create a temporary config file
    config_file = tmp_path / "test_config.yaml"
    config_content = """
cookiecutters_dir: ~/my_cookiecutters
replay_dir: ~/my_replay
default_context:
  author_name: John Doe
abbreviations:
  custom: https://custom.com/{0}.git
"""
    config_file.write_text(config_content)
    
    # Test successful config loading
    config = get_config(str(config_file))
    
    assert isinstance(config, dict)
    assert 'cookiecutters_dir' in config
    assert 'replay_dir' in config
    assert 'default_context' in config
    assert 'abbreviations' in config
    # Check that paths are expanded
    assert '~' not in config['cookiecutters_dir']
    assert '~' not in config['replay_dir']
    # Check that custom values are merged with defaults
    assert config['default_context']['author_name'] == 'John Doe'
    assert config['abbreviations']['custom'] == 'https://custom.com/{0}.git'
    # Check that builtin abbreviations are preserved
    assert 'gh' in config['abbreviations']
    assert 'gl' in config['abbreviations']
    assert 'bb' in config['abbreviations']


def test_get_config_file_not_exists():
    """Test get_config raises ConfigDoesNotExistException for non-existent file."""
    with pytest.raises(ConfigDoesNotExistException) as exc_info:
        get_config('/nonexistent/path/config.yaml')
    
    assert 'does not exist' in str(exc_info.value)


def test_get_config_invalid_yaml(tmp_path):
    """Test get_config raises InvalidConfiguration for invalid YAML."""
    config_file = tmp_path / "invalid_config.yaml"
    config_file.write_text("invalid: yaml: content: [")
    
    with pytest.raises(InvalidConfiguration) as exc_info:
        get_config(str(config_file))
    
    assert 'Unable to parse YAML file' in str(exc_info.value)


def test_get_config_non_dict_top_level(tmp_path):
    """Test get_config raises InvalidConfiguration when top-level is not a dict."""
    config_file = tmp_path / "list_config.yaml"
    config_file.write_text("- item1\n- item2\n")
    
    with pytest.raises(InvalidConfiguration) as exc_info:
        get_config(str(config_file))
    
    assert 'should be an object' in str(exc_info.value)


def test_get_config_empty_yaml(tmp_path):
    """Test get_config with empty YAML file returns default config."""
    config_file = tmp_path / "empty_config.yaml"
    config_file.write_text("")
    
    config = get_config(str(config_file))
    
    assert config['cookiecutters_dir'] == os.path.expanduser('~/.cookiecutters/')
    assert config['replay_dir'] == os.path.expanduser('~/.cookiecutter_replay/')


def test_get_config_expands_environment_variables(tmp_path, monkeypatch):
    """Test get_config expands environment variables in paths."""
    monkeypatch.setenv('MY_CUSTOM_DIR', '/custom/dir')
    
    config_file = tmp_path / "env_config.yaml"
    config_content = """
cookiecutters_dir: $MY_CUSTOM_DIR/cookiecutters
replay_dir: $MY_CUSTOM_DIR/replay
"""
    config_file.write_text(config_content)
    
    config = get_config(str(config_file))
    
    assert config['cookiecutters_dir'] == '/custom/dir/cookiecutters'
    assert config['replay_dir'] == '/custom/dir/replay'


def test_get_config_path_object(tmp_path):
    """Test get_config accepts Path objects."""
    from pathlib import Path
    
    config_file = tmp_path / "path_config.yaml"
    config_file.write_text("cookiecutters_dir: ~/test\n")
    
    config = get_config(config_file)
    
    assert isinstance(config, dict)
    assert 'cookiecutters_dir' in config


# LLM-generated content at query #12
#--------------------------

```python
def test_get_config(tmp_path, monkeypatch):
    """Test get_config function with various scenarios."""
    
    # Test 1: Config file does not exist
    with pytest.raises(ConfigDoesNotExistException):
        get_config('/nonexistent/path/config.yaml')
    
    # Test 2: Valid YAML config file with all keys
    config_file = tmp_path / "config.yaml"
    config_content = """
cookiecutters_dir: ~/my_cookiecutters
replay_dir: ~/my_replay
default_context:
    author_name: John Doe
abbreviations:
    custom: https://example.com/{0}.git
"""
    config_file.write_text(config_content)
    result = get_config(config_file)
    
    assert result['cookiecutters_dir'] == os.path.expanduser('~/my_cookiecutters')
    assert result['replay_dir'] == os.path.expanduser('~/my_replay')
    assert result['default_context']['author_name'] == 'John Doe'
    assert result['abbreviations']['custom'] == 'https://example.com/{0}.git'
    assert result['abbreviations']['gh'] == 'https://github.com/{0}.git'
    
    # Test 3: Minimal config file (only some keys)
    minimal_config = tmp_path / "minimal.yaml"
    minimal_config.write_text("cookiecutters_dir: ~/custom_cookies\n")
    result = get_config(minimal_config)
    
    assert result['cookiecutters_dir'] == os.path.expanduser('~/custom_cookies')
    assert 'replay_dir' in result
    assert result['abbreviations']['gh'] == 'https://github.com/{0}.git'
    
    # Test 4: Empty YAML file
    empty_config = tmp_path / "empty.yaml"
    empty_config.write_text("")
    result = get_config(empty_config)
    
    assert result == DEFAULT_CONFIG
    
    # Test 5: Invalid YAML syntax
    invalid_yaml = tmp_path / "invalid.yaml"
    invalid_yaml.write_text("{ invalid yaml: [")
    
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml)
    
    # Test 6: YAML file with non-dict top-level element
    non_dict_yaml = tmp_path / "non_dict.yaml"
    non_dict_yaml.write_text("- item1\n- item2\n")
    
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml)
    
    # Test 7: Path with environment variables
    env_config = tmp_path / "env_config.yaml"
    env_config.write_text("cookiecutters_dir: $HOME/.my_cookies\n")
    monkeypatch.setenv('HOME', '/home/testuser')
    result = get_config(env_config)
    
    assert '/home/testuser/.my_cookies' in result['cookiecutters_dir']
    
    # Test 8: Config with nested abbreviations (merge test)
    nested_config = tmp_path / "nested.yaml"
    nested_config.write_text("""
abbreviations:
    gh: https://github.com/custom/{0}.git
    custom_bb: https://custom.bitbucket.org/{0}
""")
    result = get_config(nested_config)
    
    assert result['abbreviations']['gh'] == 'https://github.com/custom/{0}.git'
    assert result['abbreviations']['custom_bb'] == 'https://custom.bitbucket.org/{0}'
    assert result['abbreviations']['gl'] == 'https://gitlab.com/{0}.git'
    
    # Test 9: Config with tilde expansion
    tilde_config = tmp_path / "tilde.yaml"
    tilde_config.write_text("replay_dir: ~/test_replay\n")
    result = get_config(tilde_config)
    
    assert result['replay_dir'] == os.path.expanduser('~/test_replay')
    assert '~' not in result['replay_dir']


# LLM-generated content at query #13
#--------------------------

```python
import pytest
import os
import tempfile
from pathlib import Path
import yaml


def test_get_config():
    """Test get_config function with various scenarios."""
    
    # Test 1: Config file does not exist
    with pytest.raises(ConfigDoesNotExistException):
        get_config('/nonexistent/path/config.yaml')
    
    # Test 2: Valid YAML config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        config_data = {
            'cookiecutters_dir': '~/.cookiecutters/',
            'replay_dir': '~/.cookiecutter_replay/',
        }
        yaml.dump(config_data, f)
        config_file = f.name
    
    try:
        result = get_config(config_file)
        assert isinstance(result, dict)
        assert 'cookiecutters_dir' in result
        assert 'replay_dir' in result
        assert result['cookiecutters_dir'] == os.path.expanduser('~/.cookiecutters/')
        assert result['replay_dir'] == os.path.expanduser('~/.cookiecutter_replay/')
    finally:
        os.unlink(config_file)
    
    # Test 3: Invalid YAML syntax
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write('invalid: yaml: content: [')
        config_file = f.name
    
    try:
        with pytest.raises(InvalidConfiguration):
            get_config(config_file)
    finally:
        os.unlink(config_file)
    
    # Test 4: YAML file with non-dict top-level element
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(['not', 'a', 'dict'], f)
        config_file = f.name
    
    try:
        with pytest.raises(InvalidConfiguration):
            get_config(config_file)
    finally:
        os.unlink(config_file)
    
    # Test 5: Empty YAML file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write('')
        config_file = f.name
    
    try:
        result = get_config(config_file)
        assert isinstance(result, dict)
        assert result == DEFAULT_CONFIG
    finally:
        os.unlink(config_file)
    
    # Test 6: Config with environment variables in paths
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        config_data = {
            'cookiecutters_dir': '$HOME/.cookiecutters/',
            'replay_dir': '${HOME}/.cookiecutter_replay/',
        }
        yaml.dump(config_data, f)
        config_file = f.name
    
    try:
        result = get_config(config_file)
        assert result['cookiecutters_dir'] == os.path.expanduser('~/.cookiecutters/')
        assert result['replay_dir'] == os.path.expanduser('~/.cookiecutter_replay/')
    finally:
        os.unlink(config_file)
    
    # Test 7: Config with abbreviations override
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        config_data = {
            'abbreviations': {
                'custom': 'https://custom.com/{0}.git',
            }
        }
        yaml.dump(config_data, f)
        config_file = f.name
    
    try:
        result = get_config(config_file)
        assert 'abbreviations' in result
        assert 'custom' in result['abbreviations']
        assert 'gh' in result['abbreviations']  # Built-in should be preserved
    finally:
        os.unlink(config_file)
    
    # Test 8: Config as Path object
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        config_data = {'cookiecutters_dir': '~/.cookiecutters/'}
        yaml.dump(config_data, f)
        config_file = f.name
    
    try:
        result = get_config(Path(config_file))
        assert isinstance(result, dict)
        assert 'cookiecutters_dir' in result
    finally:
        os.unlink(config_file)


# LLM-generated content at query #14
#--------------------------

```python
import os
import pytest
import tempfile
from pathlib import Path
from cookiecutter.exceptions import ConfigDoesNotExistException, InvalidConfiguration


def test_get_config():
    """Test get_config function with various scenarios."""
    
    # Test 1: Non-existent config file raises ConfigDoesNotExistException
    with pytest.raises(ConfigDoesNotExistException):
        get_config('/nonexistent/path/config.yaml')
    
    # Test 2: Valid YAML config file is loaded and merged with defaults
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write('cookiecutters_dir: /custom/cookiecutters\n')
        f.write('replay_dir: /custom/replay\n')
        temp_config = f.name
    
    try:
        config = get_config(temp_config)
        assert config['cookiecutters_dir'] == '/custom/cookiecutters'
        assert config['replay_dir'] == '/custom/replay'
        assert 'default_context' in config
        assert 'abbreviations' in config
    finally:
        os.unlink(temp_config)
    
    # Test 3: Empty YAML file returns default config
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write('')
        temp_config = f.name
    
    try:
        config = get_config(temp_config)
        assert config['cookiecutters_dir'] == os.path.expanduser('~/.cookiecutters/')
        assert config['replay_dir'] == os.path.expanduser('~/.cookiecutter_replay/')
    finally:
        os.unlink(temp_config)
    
    # Test 4: Path expansion works for environment variables
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write('cookiecutters_dir: $HOME/.custom_cookiecutters\n')
        temp_config = f.name
    
    try:
        config = get_config(temp_config)
        assert '$HOME' not in config['cookiecutters_dir']
        assert config['cookiecutters_dir'].startswith(os.path.expanduser('~'))
    finally:
        os.unlink(temp_config)
    
    # Test 5: Tilde expansion works in paths
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write('cookiecutters_dir: ~/my_cookiecutters\n')
        temp_config = f.name
    
    try:
        config = get_config(temp_config)
        assert '~' not in config['cookiecutters_dir']
        assert config['cookiecutters_dir'].startswith(os.path.expanduser('~'))
    finally:
        os.unlink(temp_config)
    
    # Test 6: Invalid YAML raises InvalidConfiguration
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write('invalid: yaml: content: [')
        temp_config = f.name
    
    try:
        with pytest.raises(InvalidConfiguration):
            get_config(temp_config)
    finally:
        os.unlink(temp_config)
    
    # Test 7: Non-dict top-level YAML raises InvalidConfiguration
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write('- item1\n- item2\n')
        temp_config = f.name
    
    try:
        with pytest.raises(InvalidConfiguration):
            get_config(temp_config)
    finally:
        os.unlink(temp_config)
    
    # Test 8: Config with abbreviations merges with defaults
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write('abbreviations:\n')
        f.write('  custom: https://custom.com/{0}.git\n')
        temp_config = f.name
    
    try:
        config = get_config(temp_config)
        assert 'gh' in config['abbreviations']
        assert 'custom' in config['abbreviations']
        assert config['abbreviations']['custom'] == 'https://custom.com/{0}.git'
    finally:
        os.unlink(temp_config)
    
    # Test 9: Config accepts Path object
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write('cookiecutters_dir: /test/path\n')
        temp_config = f.name
    
    try:
        config = get_config(Path(temp_config))
        assert config['cookiecutters_dir'] == '/test/path'
    finally:
        os.unlink(temp_config)


# LLM-generated content at query #15
#--------------------------

```python
def test_get_config(tmp_path, monkeypatch):
    """Test get_config function with various scenarios."""
    # Test 1: Non-existent config file raises ConfigDoesNotExistException
    with pytest.raises(ConfigDoesNotExistException):
        get_config('/non/existent/path/config.yaml')

    # Test 2: Valid YAML config file is loaded and merged with defaults
    config_file = tmp_path / 'config.yaml'
    config_file.write_text(
        'cookiecutters_dir: /custom/cookiecutters\n'
        'replay_dir: /custom/replay\n'
    )
    result = get_config(config_file)
    assert result['cookiecutters_dir'] == '/custom/cookiecutters'
    assert result['replay_dir'] == '/custom/replay'
    assert 'abbreviations' in result
    assert result['abbreviations'] == BUILTIN_ABBREVIATIONS

    # Test 3: Config with environment variable expansion
    config_file = tmp_path / 'config_env.yaml'
    config_file.write_text('cookiecutters_dir: $HOME/.cookiecutters\n')
    result = get_config(config_file)
    assert result['cookiecutters_dir'] == os.path.expanduser('~/.cookiecutters')

    # Test 4: Config with tilde expansion
    config_file = tmp_path / 'config_tilde.yaml'
    config_file.write_text('replay_dir: ~/my_replay\n')
    result = get_config(config_file)
    assert result['replay_dir'] == os.path.expanduser('~/my_replay')

    # Test 5: Empty YAML file returns defaults
    config_file = tmp_path / 'empty_config.yaml'
    config_file.write_text('')
    result = get_config(config_file)
    assert result == DEFAULT_CONFIG

    # Test 6: Invalid YAML raises InvalidConfiguration
    config_file = tmp_path / 'invalid_config.yaml'
    config_file.write_text('invalid: yaml: content: [')
    with pytest.raises(InvalidConfiguration):
        get_config(config_file)

    # Test 7: Non-dict top-level YAML raises InvalidConfiguration
    config_file = tmp_path / 'list_config.yaml'
    config_file.write_text('- item1\n- item2\n')
    with pytest.raises(InvalidConfiguration):
        get_config(config_file)

    # Test 8: Nested dict merging preserves defaults
    config_file = tmp_path / 'nested_config.yaml'
    config_file.write_text(
        'abbreviations:\n'
        '  custom: https://custom.com/{0}\n'
    )
    result = get_config(config_file)
    assert 'gh' in result['abbreviations']
    assert 'custom' in result['abbreviations']
    assert result['abbreviations']['gh'] == BUILTIN_ABBREVIATIONS['gh']

    # Test 9: Config file as Path object
    config_file = tmp_path / 'path_config.yaml'
    config_file.write_text('cookiecutters_dir: /path/cookiecutters\n')
    result = get_config(config_file)
    assert result['cookiecutters_dir'] == '/path/cookiecutters'

    # Test 10: Config file as string path
    config_str = str(tmp_path / 'str_config.yaml')
    with open(config_str, 'w') as f:
        f.write('replay_dir: /path/replay\n')
    result = get_config(config_str)
    assert result['replay_dir'] == '/path/replay'


# LLM-generated content at query #16
#--------------------------

```python
def test_get_config(tmp_path):
    """Test get_config function with valid and invalid config files."""
    # Test with non-existent config file
    non_existent = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent)

    # Test with valid YAML config file
    valid_config = tmp_path / "valid_config.yaml"
    valid_config.write_text(
        "cookiecutters_dir: /tmp/cookies\n"
        "replay_dir: /tmp/replay\n"
        "abbreviations:\n"
        "  custom: https://example.com/{0}.git\n"
    )
    config = get_config(valid_config)
    assert config['cookiecutters_dir'] == '/tmp/cookies'
    assert config['replay_dir'] == '/tmp/replay'
    assert 'custom' in config['abbreviations']
    assert config['abbreviations']['gh'] == 'https://github.com/{0}.git'

    # Test with empty YAML file
    empty_config = tmp_path / "empty_config.yaml"
    empty_config.write_text("")
    config = get_config(empty_config)
    assert config == DEFAULT_CONFIG

    # Test with invalid YAML syntax
    invalid_yaml = tmp_path / "invalid.yaml"
    invalid_yaml.write_text("invalid: yaml: content:")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml)

    # Test with non-dict top-level element
    non_dict_config = tmp_path / "non_dict.yaml"
    non_dict_config.write_text("- item1\n- item2\n")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_config)

    # Test path expansion with environment variables
    env_config = tmp_path / "env_config.yaml"
    env_config.write_text("cookiecutters_dir: $HOME/.cookiecutters\n")
    config = get_config(env_config)
    assert "$HOME" not in config['cookiecutters_dir']
    assert config['cookiecutters_dir'].startswith(os.path.expanduser("~"))

    # Test path expansion with tilde
    tilde_config = tmp_path / "tilde_config.yaml"
    tilde_config.write_text("replay_dir: ~/my_replay\n")
    config = get_config(tilde_config)
    assert "~" not in config['replay_dir']
    assert config['replay_dir'].startswith(os.path.expanduser("~"))

    # Test with default_context
    context_config = tmp_path / "context_config.yaml"
    context_config.write_text(
        "default_context:\n"
        "  project_name: my_project\n"
        "  author: John Doe\n"
    )
    config = get_config(context_config)
    assert config['default_context']['project_name'] == 'my_project'
    assert config['default_context']['author'] == 'John Doe'

    # Test merging with defaults
    partial_config = tmp_path / "partial_config.yaml"
    partial_config.write_text("cookiecutters_dir: /custom/path\n")
    config = get_config(partial_config)
    assert config['cookiecutters_dir'] == '/custom/path'
    assert 'replay_dir' in config
    assert config['abbreviations']['bb'] == 'https://bitbucket.org/{0}'


# LLM-generated content at query #17
#--------------------------

```python
def test_get_config(tmp_path):
    """Test get_config function with various scenarios."""
    
    # Test 1: Config file does not exist
    with pytest.raises(ConfigDoesNotExistException):
        get_config(tmp_path / 'nonexistent.yaml')
    
    # Test 2: Valid config file with basic content
    config_file = tmp_path / 'valid_config.yaml'
    config_file.write_text('cookiecutters_dir: /custom/path\n')
    result = get_config(config_file)
    assert result['cookiecutters_dir'] == '/custom/path'
    assert 'replay_dir' in result
    assert 'default_context' in result
    
    # Test 3: Config file with environment variable expansion
    config_file = tmp_path / 'env_config.yaml'
    config_file.write_text('cookiecutters_dir: $HOME/.custom_cookiecutters\n')
    result = get_config(config_file)
    assert '$HOME' not in result['cookiecutters_dir']
    
    # Test 4: Config file with user home expansion
    config_file = tmp_path / 'home_config.yaml'
    config_file.write_text('replay_dir: ~/.custom_replay\n')
    result = get_config(config_file)
    assert '~' not in result['replay_dir']
    
    # Test 5: Config file with abbreviations merge
    config_file = tmp_path / 'abbrev_config.yaml'
    config_file.write_text('abbreviations:\n  custom: https://custom.com/{0}\n')
    result = get_config(config_file)
    assert 'gh' in result['abbreviations']
    assert 'custom' in result['abbreviations']
    assert result['abbreviations']['custom'] == 'https://custom.com/{0}'
    
    # Test 6: Invalid YAML file
    config_file = tmp_path / 'invalid.yaml'
    config_file.write_text('invalid: yaml: content: ][')
    with pytest.raises(InvalidConfiguration):
        get_config(config_file)
    
    # Test 7: YAML file with non-dict top-level element
    config_file = tmp_path / 'list_config.yaml'
    config_file.write_text('- item1\n- item2\n')
    with pytest.raises(InvalidConfiguration):
        get_config(config_file)
    
    # Test 8: Empty YAML file
    config_file = tmp_path / 'empty_config.yaml'
    config_file.write_text('')
    result = get_config(config_file)
    assert result == DEFAULT_CONFIG
    
    # Test 9: Config with default_context
    config_file = tmp_path / 'context_config.yaml'
    config_file.write_text('default_context:\n  project_name: myproject\n  author: John Doe\n')
    result = get_config(config_file)
    assert result['default_context']['project_name'] == 'myproject'
    assert result['default_context']['author'] == 'John Doe'
    
    # Test 10: Config with multiple paths
    config_file = tmp_path / 'multi_path_config.yaml'
    config_file.write_text(
        'cookiecutters_dir: /path1\n'
        'replay_dir: /path2\n'
        'abbreviations:\n'
        '  custom1: https://custom1.com/{0}\n'
    )
    result = get_config(config_file)
    assert result['cookiecutters_dir'] == '/path1'
    assert result['replay_dir'] == '/path2'
    assert 'custom1' in result['abbreviations']


# LLM-generated content at query #18
#--------------------------

```python
def test_get_config(tmp_path, monkeypatch):
    """Test get_config function with various scenarios."""
    
    # Test 1: Config file does not exist
    non_existent_path = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_path)
    
    # Test 2: Valid YAML config file
    valid_config_file = tmp_path / "valid_config.yaml"
    valid_config_content = """
cookiecutters_dir: /custom/cookiecutters
replay_dir: /custom/replay
default_context:
    author_name: Test Author
abbreviations:
    custom: https://example.com/{0}.git
"""
    valid_config_file.write_text(valid_config_content, encoding='utf-8')
    config = get_config(valid_config_file)
    
    assert config['cookiecutters_dir'] == '/custom/cookiecutters'
    assert config['replay_dir'] == '/custom/replay'
    assert config['default_context']['author_name'] == 'Test Author'
    assert config['abbreviations']['custom'] == 'https://example.com/{0}.git'
    assert config['abbreviations']['gh'] == 'https://github.com/{0}.git'
    
    # Test 3: Path expansion with environment variables
    env_var_config_file = tmp_path / "env_config.yaml"
    monkeypatch.setenv('TEST_DIR', str(tmp_path))
    env_config_content = """
cookiecutters_dir: $TEST_DIR/cookiecutters
replay_dir: ~/custom_replay
"""
    env_var_config_file.write_text(env_config_content, encoding='utf-8')
    config = get_config(env_var_config_file)
    
    assert str(tmp_path) in config['cookiecutters_dir']
    assert '~' not in config['replay_dir']
    
    # Test 4: Invalid YAML syntax
    invalid_yaml_file = tmp_path / "invalid.yaml"
    invalid_yaml_file.write_text("invalid: yaml: content: [", encoding='utf-8')
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)
    
    # Test 5: YAML file with non-dict top-level element
    invalid_dict_file = tmp_path / "invalid_dict.yaml"
    invalid_dict_file.write_text("- item1\n- item2\n", encoding='utf-8')
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_dict_file)
    
    # Test 6: Empty YAML file
    empty_yaml_file = tmp_path / "empty.yaml"
    empty_yaml_file.write_text("", encoding='utf-8')
    config = get_config(empty_yaml_file)
    
    assert config == DEFAULT_CONFIG
    
    # Test 7: Partial config with defaults merged
    partial_config_file = tmp_path / "partial.yaml"
    partial_config_file.write_text("cookiecutters_dir: /partial/path\n", encoding='utf-8')
    config = get_config(partial_config_file)
    
    assert config['cookiecutters_dir'] == '/partial/path'
    assert 'replay_dir' in config
    assert 'default_context' in config
    assert 'abbreviations' in config
    
    # Test 8: Config with tilde expansion
    tilde_config_file = tmp_path / "tilde_config.yaml"
    tilde_config_file.write_text("cookiecutters_dir: ~/my_cookiecutters\n", encoding='utf-8')
    config = get_config(tilde_config_file)
    
    assert '~' not in config['cookiecutters_dir']
    assert config['cookiecutters_dir'].startswith(os.path.expanduser('~'))


# LLM-generated content at query #19
#--------------------------

```python
def test_get_config(tmp_path, monkeypatch):
    """Test get_config function with various scenarios."""
    
    # Test 1: Config file does not exist
    with pytest.raises(ConfigDoesNotExistException):
        get_config('/nonexistent/path/config.yaml')
    
    # Test 2: Valid config file with basic content
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "cookiecutters_dir: /tmp/cookies\n"
        "replay_dir: /tmp/replay\n"
    )
    result = get_config(config_file)
    assert result['cookiecutters_dir'] == '/tmp/cookies'
    assert result['replay_dir'] == '/tmp/replay'
    assert 'abbreviations' in result
    
    # Test 3: Config with environment variable expansion
    config_file2 = tmp_path / "config2.yaml"
    config_file2.write_text("cookiecutters_dir: $HOME/.cookiecutters\n")
    monkeypatch.setenv('HOME', '/home/testuser')
    result = get_config(config_file2)
    assert result['cookiecutters_dir'] == '/home/testuser/.cookiecutters'
    
    # Test 4: Config with tilde expansion
    config_file3 = tmp_path / "config3.yaml"
    config_file3.write_text("replay_dir: ~/my_replay\n")
    result = get_config(config_file3)
    assert '~' not in result['replay_dir']
    assert result['replay_dir'].startswith(os.path.expanduser('~'))
    
    # Test 5: Invalid YAML syntax
    config_file4 = tmp_path / "config4.yaml"
    config_file4.write_text("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(config_file4)
    
    # Test 6: Non-dict top-level element in YAML
    config_file5 = tmp_path / "config5.yaml"
    config_file5.write_text("- item1\n- item2\n")
    with pytest.raises(InvalidConfiguration):
        get_config(config_file5)
    
    # Test 7: Empty config file
    config_file6 = tmp_path / "config6.yaml"
    config_file6.write_text("")
    result = get_config(config_file6)
    assert result == DEFAULT_CONFIG
    
    # Test 8: Config with abbreviations merge
    config_file7 = tmp_path / "config7.yaml"
    config_file7.write_text(
        "abbreviations:\n"
        "  custom: https://custom.com/{0}\n"
    )
    result = get_config(config_file7)
    assert 'gh' in result['abbreviations']
    assert 'custom' in result['abbreviations']
    assert result['abbreviations']['custom'] == 'https://custom.com/{0}'
    
    # Test 9: Config with default_context
    config_file8 = tmp_path / "config8.yaml"
    config_file8.write_text(
        "default_context:\n"
        "  author_name: John Doe\n"
    )
    result = get_config(config_file8)
    assert 'author_name' in result['default_context']
    assert result['default_context']['author_name'] == 'John Doe'
    
    # Test 10: Nested path with multiple expansions
    config_file9 = tmp_path / "config9.yaml"
    config_file9.write_text(
        "cookiecutters_dir: $HOME/test/~/.cookiecutters\n"
    )
    monkeypatch.setenv('HOME', '/home/user')
    result = get_config(config_file9)
    assert '$HOME' not in result['cookiecutters_dir']


# LLM-generated content at query #20
#--------------------------

```python
def test_get_config(tmp_path, monkeypatch):
    """Test get_config function with valid and invalid configurations."""
    # Test 1: Valid YAML config file
    config_file = tmp_path / "valid_config.yaml"
    config_file.write_text(
        "cookiecutters_dir: /tmp/test\n"
        "replay_dir: /tmp/replay\n"
        "default_context:\n"
        "  author: Test Author\n"
        "abbreviations:\n"
        "  custom: https://example.com/{0}.git\n"
    )
    
    result = get_config(config_file)
    
    assert result['cookiecutters_dir'] == '/tmp/test'
    assert result['replay_dir'] == '/tmp/replay'
    assert result['default_context']['author'] == 'Test Author'
    assert result['abbreviations']['gh'] == 'https://github.com/{0}.git'
    assert result['abbreviations']['custom'] == 'https://example.com/{0}.git'
    
    # Test 2: Config file with environment variable expansion
    config_file2 = tmp_path / "env_config.yaml"
    monkeypatch.setenv('TEST_DIR', '/expanded/path')
    config_file2.write_text("cookiecutters_dir: $TEST_DIR/cookies\n")
    
    result2 = get_config(config_file2)
    assert result2['cookiecutters_dir'] == '/expanded/path/cookies'
    
    # Test 3: Config file with user home expansion
    config_file3 = tmp_path / "home_config.yaml"
    config_file3.write_text("cookiecutters_dir: ~/my_cookies\n")
    
    result3 = get_config(config_file3)
    assert '~' not in result3['cookiecutters_dir']
    assert result3['cookiecutters_dir'].startswith(os.path.expanduser('~'))
    
    # Test 4: Non-existent config file
    non_existent = tmp_path / "does_not_exist.yaml"
    
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent)
    
    # Test 5: Invalid YAML syntax
    invalid_yaml = tmp_path / "invalid.yaml"
    invalid_yaml.write_text("invalid: yaml: content: [\n")
    
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml)
    
    # Test 6: YAML file with non-dict top-level element
    non_dict_yaml = tmp_path / "non_dict.yaml"
    non_dict_yaml.write_text("- item1\n- item2\n")
    
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml)
    
    # Test 7: Empty YAML file
    empty_yaml = tmp_path / "empty.yaml"
    empty_yaml.write_text("")
    
    result7 = get_config(empty_yaml)
    assert result7['cookiecutters_dir'] == os.path.expanduser('~/.cookiecutters/')
    assert result7['replay_dir'] == os.path.expanduser('~/.cookiecutter_replay/')
    
    # Test 8: Partial config merges with defaults
    partial_config = tmp_path / "partial.yaml"
    partial_config.write_text("default_context:\n  name: TestProject\n")
    
    result8 = get_config(partial_config)
    assert result8['default_context']['name'] == 'TestProject'
    assert result8['cookiecutters_dir'] == os.path.expanduser('~/.cookiecutters/')
    assert 'gh' in result8['abbreviations']


# LLM-generated content at query #21
#--------------------------

```python
def test_get_config(tmp_path, monkeypatch):
    """Test get_config function with various scenarios."""
    
    # Test 1: Config file does not exist
    non_existent_path = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_path)
    
    # Test 2: Valid config file with basic content
    config_file = tmp_path / "valid_config.yaml"
    config_content = {
        'cookiecutters_dir': '/custom/path',
        'default_context': {'key': 'value'},
    }
    config_file.write_text(yaml.dump(config_content))
    
    result = get_config(config_file)
    assert result['cookiecutters_dir'] == '/custom/path'
    assert result['default_context']['key'] == 'value'
    assert result['abbreviations'] == BUILTIN_ABBREVIATIONS
    
    # Test 3: Config with environment variable expansion
    config_file = tmp_path / "env_config.yaml"
    monkeypatch.setenv('TEST_DIR', '/expanded/dir')
    config_content = {
        'cookiecutters_dir': '$TEST_DIR/cookiecutters',
    }
    config_file.write_text(yaml.dump(config_content))
    
    result = get_config(config_file)
    assert result['cookiecutters_dir'] == '/expanded/dir/cookiecutters'
    
    # Test 4: Config with user home expansion
    config_file = tmp_path / "home_config.yaml"
    config_content = {
        'replay_dir': '~/my_replay',
    }
    config_file.write_text(yaml.dump(config_content))
    
    result = get_config(config_file)
    assert '~' not in result['replay_dir']
    assert result['replay_dir'].startswith(os.path.expanduser('~'))
    
    # Test 5: Invalid YAML syntax
    config_file = tmp_path / "invalid.yaml"
    config_file.write_text("invalid: yaml: content: [")
    
    with pytest.raises(InvalidConfiguration):
        get_config(config_file)
    
    # Test 6: YAML file with non-dict top-level element
    config_file = tmp_path / "non_dict.yaml"
    config_file.write_text("- item1\n- item2")
    
    with pytest.raises(InvalidConfiguration):
        get_config(config_file)
    
    # Test 7: Empty YAML file
    config_file = tmp_path / "empty.yaml"
    config_file.write_text("")
    
    result = get_config(config_file)
    assert result == DEFAULT_CONFIG
    
    # Test 8: Config with nested abbreviations merge
    config_file = tmp_path / "abbrev_config.yaml"
    config_content = {
        'abbreviations': {'custom': 'https://custom.com/{0}'},
    }
    config_file.write_text(yaml.dump(config_content))
    
    result = get_config(config_file)
    assert result['abbreviations']['gh'] == 'https://github.com/{0}.git'
    assert result['abbreviations']['custom'] == 'https://custom.com/{0}'
    
    # Test 9: Config file as string path
    config_file = tmp_path / "string_path.yaml"
    config_content = {'cookiecutters_dir': '/test'}
    config_file.write_text(yaml.dump(config_content))
    
    result = get_config(str(config_file))
    assert result['cookiecutters_dir'] == '/test'
    
    # Test 10: Config with both replay_dir and cookiecutters_dir expansion
    config_file = tmp_path / "both_dirs.yaml"
    config_content = {
        'cookiecutters_dir': '~/cookies',
        'replay_dir': '~/replay',
    }
    config_file.write_text(yaml.dump(config_content))
    
    result = get_config(config_file)
    assert '~' not in result['cookiecutters_dir']
    assert '~' not in result['replay_dir']
    assert result['cookiecutters_dir'].startswith(os.path.expanduser('~'))
    assert result['replay_dir'].startswith(os.path.expanduser('~'))


# LLM-generated content at query #22
#--------------------------

```python
def test_get_config(tmp_path, monkeypatch):
    """Test get_config function with various scenarios."""
    # Test 1: Config file does not exist
    non_existent_path = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_path)

    # Test 2: Valid config file with basic content
    config_file = tmp_path / "valid_config.yaml"
    config_content = {
        'cookiecutters_dir': '~/my_cookiecutters',
        'replay_dir': '~/my_replay',
        'default_context': {'project_name': 'my_project'},
        'abbreviations': {'custom': 'https://example.com/{0}'}
    }
    config_file.write_text(yaml.dump(config_content))
    
    result = get_config(config_file)
    
    assert result['cookiecutters_dir'] == os.path.expanduser('~/my_cookiecutters')
    assert result['replay_dir'] == os.path.expanduser('~/my_replay')
    assert result['default_context']['project_name'] == 'my_project'
    assert result['abbreviations']['custom'] == 'https://example.com/{0}'
    assert result['abbreviations']['gh'] == 'https://github.com/{0}.git'

    # Test 3: Config file with environment variable expansion
    config_file_env = tmp_path / "config_with_env.yaml"
    monkeypatch.setenv('TEST_DIR', '/test/path')
    config_content_env = {
        'cookiecutters_dir': '$TEST_DIR/cookiecutters',
        'replay_dir': '$TEST_DIR/replay'
    }
    config_file_env.write_text(yaml.dump(config_content_env))
    
    result_env = get_config(config_file_env)
    
    assert result_env['cookiecutters_dir'] == '/test/path/cookiecutters'
    assert result_env['replay_dir'] == '/test/path/replay'

    # Test 4: Invalid YAML syntax
    invalid_yaml_file = tmp_path / "invalid.yaml"
    invalid_yaml_file.write_text("{ invalid yaml content: [")
    
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test 5: YAML file with non-dict top-level element
    non_dict_file = tmp_path / "non_dict.yaml"
    non_dict_file.write_text("- item1\n- item2")
    
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_file)

    # Test 6: Empty config file
    empty_config_file = tmp_path / "empty_config.yaml"
    empty_config_file.write_text("")
    
    result_empty = get_config(empty_config_file)
    
    assert result_empty['cookiecutters_dir'] == DEFAULT_CONFIG['cookiecutters_dir']
    assert result_empty['replay_dir'] == DEFAULT_CONFIG['replay_dir']
    assert result_empty['abbreviations'] == BUILTIN_ABBREVIATIONS

    # Test 7: Partial config merging with defaults
    partial_config_file = tmp_path / "partial_config.yaml"
    partial_content = {'cookiecutters_dir': '~/custom_dir'}
    partial_config_file.write_text(yaml.dump(partial_content))
    
    result_partial = get_config(partial_config_file)
    
    assert result_partial['cookiecutters_dir'] == os.path.expanduser('~/custom_dir')
    assert result_partial['replay_dir'] == DEFAULT_CONFIG['replay_dir']
    assert result_partial['abbreviations'] == BUILTIN_ABBREVIATIONS

    # Test 8: Config with tilde expansion
    tilde_config_file = tmp_path / "tilde_config.yaml"
    tilde_content = {
        'cookiecutters_dir': '~/.my_cookiecutters',
        'replay_dir': '~/.my_replay'
    }
    tilde_config_file.write_text(yaml.dump(tilde_content))
    
    result_tilde = get_config(tilde_config_file)
    
    assert result_tilde['cookiecutters_dir'] == os.path.expanduser('~/.my_cookiecutters')
    assert result_tilde['replay_dir'] == os.path.expanduser('~/.my_replay')


# LLM-generated content at query #23
#--------------------------

```python
def test_get_user_config(tmp_path, monkeypatch):
    """Test get_user_config function with various scenarios."""
    
    # Test 1: Return default config when default_config=True
    result = get_user_config(default_config=True)
    assert result == DEFAULT_CONFIG
    assert isinstance(result, dict)
    assert 'cookiecutters_dir' in result
    assert 'replay_dir' in result
    
    # Test 2: Merge provided dict with defaults when default_config is a dict
    custom_defaults = {'cookiecutters_dir': '/custom/path'}
    result = get_user_config(default_config=custom_defaults)
    assert result['cookiecutters_dir'] == '/custom/path'
    assert 'replay_dir' in result
    assert result['replay_dir'] == DEFAULT_CONFIG['replay_dir']
    
    # Test 3: Load custom config file
    config_file = tmp_path / 'custom_config.yaml'
    config_content = """
cookiecutters_dir: /tmp/custom_cookiecutters
replay_dir: /tmp/custom_replay
"""
    config_file.write_text(config_content)
    result = get_user_config(config_file=str(config_file))
    assert result['cookiecutters_dir'] == '/tmp/custom_cookiecutters'
    assert result['replay_dir'] == '/tmp/custom_replay'
    
    # Test 4: Load config from COOKIECUTTER_CONFIG environment variable
    env_config_file = tmp_path / 'env_config.yaml'
    env_config_content = """
cookiecutters_dir: /tmp/env_cookiecutters
"""
    env_config_file.write_text(env_config_content)
    monkeypatch.setenv('COOKIECUTTER_CONFIG', str(env_config_file))
    monkeypatch.delenv('COOKIECUTTER_CONFIG', raising=False)
    monkeypatch.setenv('COOKIECUTTER_CONFIG', str(env_config_file))
    result = get_user_config()
    assert result['cookiecutters_dir'] == '/tmp/env_cookiecutters'
    
    # Test 5: Return default config when COOKIECUTTER_CONFIG env var not set and user config doesn't exist
    monkeypatch.delenv('COOKIECUTTER_CONFIG', raising=False)
    monkeypatch.setattr('cookiecutter.config.USER_CONFIG_PATH', str(tmp_path / 'nonexistent.yaml'))
    result = get_user_config()
    assert result == DEFAULT_CONFIG
    
    # Test 6: Load user config from default path if it exists
    user_config_path = tmp_path / '.cookiecutterrc'
    user_config_content = """
cookiecutters_dir: /tmp/user_cookiecutters
"""
    user_config_path.write_text(user_config_content)
    monkeypatch.setattr('cookiecutter.config.USER_CONFIG_PATH', str(user_config_path))
    monkeypatch.delenv('COOKIECUTTER_CONFIG', raising=False)
    result = get_user_config()
    assert result['cookiecutters_dir'] == '/tmp/user_cookiecutters'
    
    # Test 7: ConfigDoesNotExistException when config_file doesn't exist
    from cookiecutter.exceptions import ConfigDoesNotExistException
    with pytest.raises(ConfigDoesNotExistException):
        get_user_config(config_file='/nonexistent/path/config.yaml')
    
    # Test 8: InvalidConfiguration when YAML is invalid
    invalid_config_file = tmp_path / 'invalid_config.yaml'
    invalid_config_file.write_text('{ invalid yaml content: [')
    from cookiecutter.exceptions import InvalidConfiguration
    with pytest.raises(InvalidConfiguration):
        get_user_config(config_file=str(invalid_config_file))
    
    # Test 9: default_config=True takes precedence over config_file
    config_file = tmp_path / 'config.yaml'
    config_file.write_text('cookiecutters_dir: /should/be/ignored')
    result = get_user_config(config_file=str(config_file), default_config=True)
    assert result == DEFAULT_CONFIG
    
    # Test 10: Merge configs properly preserves abbreviations
    config_with_abbrev = tmp_path / 'abbrev_config.yaml'
    config_with_abbrev.write_text("""
abbreviations:
  custom: https://custom.com/{0}
""")
    result = get_user_config(config_file=str(config_with_abbrev))
    assert 'custom' in result['abbreviations']
    assert 'gh' in result['abbreviations']
    assert result['abbreviations']['gh'] == BUILTIN_ABBREVIATIONS['gh']


# LLM-generated content at query #24
#--------------------------

```python
def test_get_config(tmp_path, monkeypatch):
    """Test get_config function with various scenarios."""
    # Test 1: Valid YAML config file
    config_file = tmp_path / "test_config.yaml"
    config_content = """
cookiecutters_dir: /custom/cookiecutters
replay_dir: /custom/replay
default_context:
    author_name: Test Author
abbreviations:
    custom: https://custom.com/{0}.git
"""
    config_file.write_text(config_content, encoding='utf-8')
    
    result = get_config(config_file)
    
    assert result['cookiecutters_dir'] == '/custom/cookiecutters'
    assert result['replay_dir'] == '/custom/replay'
    assert result['default_context']['author_name'] == 'Test Author'
    assert result['abbreviations']['custom'] == 'https://custom.com/{0}.git'
    # Built-in abbreviations should still be present
    assert result['abbreviations']['gh'] == 'https://github.com/{0}.git'

    # Test 2: Config file with environment variables
    config_file2 = tmp_path / "test_config2.yaml"
    monkeypatch.setenv('TEST_DIR', '/test/path')
    config_content2 = "cookiecutters_dir: $TEST_DIR/cookiecutters"
    config_file2.write_text(config_content2, encoding='utf-8')
    
    result2 = get_config(config_file2)
    assert result2['cookiecutters_dir'] == '/test/path/cookiecutters'

    # Test 3: Config file with tilde expansion
    config_file3 = tmp_path / "test_config3.yaml"
    config_content3 = "cookiecutters_dir: ~/my_cookiecutters"
    config_file3.write_text(config_content3, encoding='utf-8')
    
    result3 = get_config(config_file3)
    assert '~' not in result3['cookiecutters_dir']
    assert result3['cookiecutters_dir'].startswith(os.path.expanduser('~'))

    # Test 4: Non-existent config file
    non_existent = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent)

    # Test 5: Invalid YAML syntax
    invalid_config = tmp_path / "invalid.yaml"
    invalid_config.write_text("{ invalid yaml: [", encoding='utf-8')
    
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_config)

    # Test 6: Top-level element is not a dict
    non_dict_config = tmp_path / "non_dict.yaml"
    non_dict_config.write_text("- item1\n- item2", encoding='utf-8')
    
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_config)

    # Test 7: Empty YAML file
    empty_config = tmp_path / "empty.yaml"
    empty_config.write_text("", encoding='utf-8')
    
    result7 = get_config(empty_config)
    assert result7 == DEFAULT_CONFIG

    # Test 8: Config with only some keys specified
    partial_config = tmp_path / "partial.yaml"
    partial_config.write_text("default_context:\n    key1: value1", encoding='utf-8')
    
    result8 = get_config(partial_config)
    assert 'cookiecutters_dir' in result8
    assert 'replay_dir' in result8
    assert result8['default_context']['key1'] == 'value1'


# LLM-generated content at query #25
#--------------------------

```python
def test_get_user_config(monkeypatch, tmp_path):
    """Test get_user_config function with various scenarios."""
    
    # Test 1: Return default config when default_config=True
    result = get_user_config(default_config=True)
    assert result == DEFAULT_CONFIG
    assert isinstance(result, dict)
    assert 'cookiecutters_dir' in result
    assert 'replay_dir' in result
    assert 'default_context' in result
    assert 'abbreviations' in result
    
    # Test 2: Merge provided dict with defaults when default_config is a dict
    custom_defaults = {'cookiecutters_dir': '/custom/path'}
    result = get_user_config(default_config=custom_defaults)
    assert result['cookiecutters_dir'] == '/custom/path'
    assert result['replay_dir'] == DEFAULT_CONFIG['replay_dir']
    assert 'abbreviations' in result
    
    # Test 3: Load custom config file
    config_file = tmp_path / "custom_config.yml"
    config_file.write_text("cookiecutters_dir: /tmp/cookiecutters\n")
    result = get_user_config(config_file=str(config_file))
    assert result['cookiecutters_dir'] == '/tmp/cookiecutters'
    
    # Test 4: Return defaults when no config file exists and no env var set
    monkeypatch.delenv('COOKIECUTTER_CONFIG', raising=False)
    monkeypatch.setattr(os.path, 'exists', lambda x: False)
    result = get_user_config()
    assert result == DEFAULT_CONFIG
    
    # Test 5: Load from COOKIECUTTER_CONFIG environment variable
    env_config_file = tmp_path / "env_config.yml"
    env_config_file.write_text("replay_dir: /tmp/replay\n")
    monkeypatch.setenv('COOKIECUTTER_CONFIG', str(env_config_file))
    monkeypatch.setattr(os.path, 'exists', lambda x: False)
    result = get_user_config()
    assert result['replay_dir'] == '/tmp/replay'
    
    # Test 6: Load from USER_CONFIG_PATH when it exists
    monkeypatch.delenv('COOKIECUTTER_CONFIG', raising=False)
    user_config = tmp_path / "user_config.yml"
    user_config.write_text("default_context:\n  project_name: test\n")
    
    def mock_exists(path):
        return str(path) == str(user_config) or path == str(user_config)
    
    monkeypatch.setattr(os.path, 'exists', mock_exists)
    monkeypatch.setattr('cookiecutter.config.USER_CONFIG_PATH', str(user_config))
    result = get_user_config()
    assert 'default_context' in result
    
    # Test 7: default_config=True takes precedence over config_file
    result = get_user_config(config_file=str(config_file), default_config=True)
    assert result == DEFAULT_CONFIG
    
    # Test 8: config_file takes precedence over USER_CONFIG_PATH
    monkeypatch.delenv('COOKIECUTTER_CONFIG', raising=False)
    monkeypatch.setattr(os.path, 'exists', lambda x: True)
    result = get_user_config(config_file=str(config_file))
    assert result['cookiecutters_dir'] == '/tmp/cookiecutters'
    
    # Test 9: Invalid config file raises exception
    invalid_config = tmp_path / "invalid.yml"
    invalid_config.write_text("{ invalid yaml content: [")
    with pytest.raises(InvalidConfiguration):
        get_user_config(config_file=str(invalid_config))
    
    # Test 10: Non-existent config file raises exception
    with pytest.raises(ConfigDoesNotExistException):
        get_user_config(config_file="/nonexistent/config.yml")


# LLM-generated content at query #26
#--------------------------

```python
def test_get_config(tmp_path, monkeypatch):
    """Test get_config function with various scenarios."""
    # Test 1: Valid YAML config file
    config_file = tmp_path / "config.yaml"
    config_content = """
cookiecutters_dir: /custom/cookiecutters
replay_dir: /custom/replay
default_context:
    author_name: Test Author
abbreviations:
    custom: https://custom.com/{0}.git
"""
    config_file.write_text(config_content)
    
    result = get_config(config_file)
    
    assert result['cookiecutters_dir'] == '/custom/cookiecutters'
    assert result['replay_dir'] == '/custom/replay'
    assert result['default_context']['author_name'] == 'Test Author'
    assert result['abbreviations']['custom'] == 'https://custom.com/{0}.git'
    # Built-in abbreviations should be preserved
    assert result['abbreviations']['gh'] == 'https://github.com/{0}.git'


def test_get_config_nonexistent_file(tmp_path):
    """Test get_config raises exception for non-existent file."""
    nonexistent_file = tmp_path / "nonexistent.yaml"
    
    with pytest.raises(ConfigDoesNotExistException):
        get_config(nonexistent_file)


def test_get_config_invalid_yaml(tmp_path):
    """Test get_config raises exception for invalid YAML."""
    config_file = tmp_path / "invalid.yaml"
    config_file.write_text("invalid: yaml: content: [")
    
    with pytest.raises(InvalidConfiguration):
        get_config(config_file)


def test_get_config_non_dict_yaml(tmp_path):
    """Test get_config raises exception when YAML root is not a dict."""
    config_file = tmp_path / "list.yaml"
    config_file.write_text("- item1\n- item2\n")
    
    with pytest.raises(InvalidConfiguration):
        get_config(config_file)


def test_get_config_empty_file(tmp_path):
    """Test get_config with empty YAML file."""
    config_file = tmp_path / "empty.yaml"
    config_file.write_text("")
    
    result = get_config(config_file)
    
    assert result == DEFAULT_CONFIG


def test_get_config_expands_environment_variables(tmp_path, monkeypatch):
    """Test get_config expands environment variables in paths."""
    monkeypatch.setenv('TEST_DIR', '/test/path')
    config_file = tmp_path / "config.yaml"
    config_content = """
cookiecutters_dir: $TEST_DIR/cookiecutters
replay_dir: ~/replay
"""
    config_file.write_text(config_content)
    
    result = get_config(config_file)
    
    assert result['cookiecutters_dir'] == '/test/path/cookiecutters'
    assert result['replay_dir'].endswith('replay')


def test_get_config_merges_with_defaults(tmp_path):
    """Test get_config merges custom config with defaults."""
    config_file = tmp_path / "config.yaml"
    config_content = """
default_context:
    custom_key: custom_value
"""
    config_file.write_text(config_content)
    
    result = get_config(config_file)
    
    # Should have default values
    assert 'cookiecutters_dir' in result
    assert 'replay_dir' in result
    assert 'abbreviations' in result
    # Should have custom values
    assert result['default_context']['custom_key'] == 'custom_value'


# LLM-generated content at query #27
#--------------------------

```python
def test_get_user_config(monkeypatch, tmp_path):
    """Test get_user_config function with various scenarios."""
    
    # Test 1: Return default config when default_config=True
    result = get_user_config(default_config=True)
    assert result == DEFAULT_CONFIG
    assert isinstance(result, dict)
    assert 'cookiecutters_dir' in result
    assert 'replay_dir' in result
    
    # Test 2: Merge custom dict with defaults when default_config is a dict
    custom_config = {'cookiecutters_dir': '/custom/path'}
    result = get_user_config(default_config=custom_config)
    assert result['cookiecutters_dir'] == '/custom/path'
    assert 'replay_dir' in result
    assert result['replay_dir'] == DEFAULT_CONFIG['replay_dir']
    
    # Test 3: Load config from custom config_file path
    config_file = tmp_path / 'custom_config.yaml'
    config_content = "cookiecutters_dir: /my/custom/cookiecutters\n"
    config_file.write_text(config_content)
    result = get_user_config(config_file=str(config_file))
    assert '/my/custom/cookiecutters' in result['cookiecutters_dir']
    
    # Test 4: Load from COOKIECUTTER_CONFIG environment variable
    env_config_file = tmp_path / 'env_config.yaml'
    env_config_file.write_text("replay_dir: /env/replay\n")
    monkeypatch.setenv('COOKIECUTTER_CONFIG', str(env_config_file))
    monkeypatch.delenv('COOKIECUTTER_CONFIG', raising=False)
    monkeypatch.setenv('COOKIECUTTER_CONFIG', str(env_config_file))
    result = get_user_config()
    assert '/env/replay' in result['replay_dir']
    
    # Test 5: Return default config when no config file exists and no env var set
    monkeypatch.delenv('COOKIECUTTER_CONFIG', raising=False)
    monkeypatch.setenv('HOME', str(tmp_path))
    result = get_user_config()
    assert isinstance(result, dict)
    assert 'cookiecutters_dir' in result
    
    # Test 6: Load from USER_CONFIG_PATH if it exists
    user_config_dir = tmp_path / '.cookiecutterrc'
    user_config_dir.write_text("default_context:\n  author: Test Author\n")
    monkeypatch.setattr('cookiecutter.config.USER_CONFIG_PATH', str(user_config_dir))
    monkeypatch.delenv('COOKIECUTTER_CONFIG', raising=False)
    result = get_user_config()
    assert 'default_context' in result
    
    # Test 7: Config file does not exist raises exception
    monkeypatch.delenv('COOKIECUTTER_CONFIG', raising=False)
    with pytest.raises(ConfigDoesNotExistException):
        get_user_config(config_file='/nonexistent/config.yaml')
    
    # Test 8: Invalid YAML raises exception
    invalid_config_file = tmp_path / 'invalid.yaml'
    invalid_config_file.write_text("invalid: yaml: content: ][")
    with pytest.raises(InvalidConfiguration):
        get_user_config(config_file=str(invalid_config_file))
    
    # Test 9: YAML file that is not a dict raises exception
    non_dict_config_file = tmp_path / 'non_dict.yaml'
    non_dict_config_file.write_text("- item1\n- item2\n")
    with pytest.raises(InvalidConfiguration):
        get_user_config(config_file=str(non_dict_config_file))
    
    # Test 10: Paths are expanded correctly
    config_with_vars = tmp_path / 'config_vars.yaml'
    config_with_vars.write_text("cookiecutters_dir: $HOME/.cookiecutters\n")
    result = get_user_config(config_file=str(config_with_vars))
    assert '$HOME' not in result['cookiecutters_dir']


# LLM-generated content at query #28
#--------------------------

```python
import pytest
import os
import tempfile
from pathlib import Path
import yaml


def test_get_config():
    """Test get_config function with various scenarios."""
    
    # Test 1: Config file does not exist
    with pytest.raises(ConfigDoesNotExistException):
        get_config('/nonexistent/path/config.yaml')
    
    # Test 2: Valid config file with basic content
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        config_data = {
            'cookiecutters_dir': '~/.cookiecutters/',
            'replay_dir': '~/.cookiecutter_replay/',
            'default_context': {},
        }
        yaml.dump(config_data, f)
        temp_config_path = f.name
    
    try:
        result = get_config(temp_config_path)
        assert isinstance(result, dict)
        assert 'cookiecutters_dir' in result
        assert 'replay_dir' in result
        assert result['cookiecutters_dir'] == os.path.expanduser('~/.cookiecutters/')
        assert result['replay_dir'] == os.path.expanduser('~/.cookiecutter_replay/')
    finally:
        os.unlink(temp_config_path)
    
    # Test 3: Config file with environment variable expansion
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        config_data = {
            'cookiecutters_dir': '$HOME/.cookiecutters/',
        }
        yaml.dump(config_data, f)
        temp_config_path = f.name
    
    try:
        result = get_config(temp_config_path)
        assert '$HOME' not in result['cookiecutters_dir']
    finally:
        os.unlink(temp_config_path)
    
    # Test 4: Invalid YAML file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write('invalid: yaml: content: [')
        temp_config_path = f.name
    
    try:
        with pytest.raises(InvalidConfiguration):
            get_config(temp_config_path)
    finally:
        os.unlink(temp_config_path)
    
    # Test 5: YAML file with non-dict top-level element
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write('- item1\n- item2\n')
        temp_config_path = f.name
    
    try:
        with pytest.raises(InvalidConfiguration):
            get_config(temp_config_path)
    finally:
        os.unlink(temp_config_path)
    
    # Test 6: Empty YAML file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write('')
        temp_config_path = f.name
    
    try:
        result = get_config(temp_config_path)
        assert result == DEFAULT_CONFIG
    finally:
        os.unlink(temp_config_path)
    
    # Test 7: Config merges with defaults
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        config_data = {
            'abbreviations': {'custom': 'https://custom.com/{0}'},
        }
        yaml.dump(config_data, f)
        temp_config_path = f.name
    
    try:
        result = get_config(temp_config_path)
        assert 'gh' in result['abbreviations']  # From defaults
        assert 'custom' in result['abbreviations']  # From config
    finally:
        os.unlink(temp_config_path)
    
    # Test 8: Path object as config_path
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        config_data = {'default_context': {'key': 'value'}}
        yaml.dump(config_data, f)
        temp_config_path = f.name
    
    try:
        result = get_config(Path(temp_config_path))
        assert isinstance(result, dict)
        assert 'default_context' in result
    finally:
        os.unlink(temp_config_path)


# LLM-generated content at query #29
#--------------------------

```python
def test_get_user_config(tmp_path, monkeypatch):
    """Test get_user_config function with various scenarios."""
    
    # Test 1: default_config is True - should return default config
    result = get_user_config(default_config=True)
    assert result == DEFAULT_CONFIG
    assert isinstance(result, dict)
    
    # Test 2: default_config is a dict - should merge with defaults
    custom_config = {'cookiecutters_dir': '/custom/path'}
    result = get_user_config(default_config=custom_config)
    assert result['cookiecutters_dir'] == '/custom/path'
    assert result['replay_dir'] == DEFAULT_CONFIG['replay_dir']
    
    # Test 3: config_file provided - should load from that file
    config_file = tmp_path / "test_config.yaml"
    config_content = {'cookiecutters_dir': '/test/path', 'replay_dir': '/test/replay'}
    config_file.write_text(yaml.dump(config_content))
    
    result = get_user_config(config_file=str(config_file))
    assert result['cookiecutters_dir'] == '/test/path'
    assert result['replay_dir'] == '/test/replay'
    
    # Test 4: COOKIECUTTER_CONFIG environment variable set
    env_config_file = tmp_path / "env_config.yaml"
    env_config_content = {'default_context': {'name': 'test'}}
    env_config_file.write_text(yaml.dump(env_config_content))
    
    monkeypatch.setenv('COOKIECUTTER_CONFIG', str(env_config_file))
    result = get_user_config()
    assert result['default_context']['name'] == 'test'
    
    # Test 5: No config file and no environment variable - should return defaults
    monkeypatch.delenv('COOKIECUTTER_CONFIG', raising=False)
    monkeypatch.setattr('os.path.exists', lambda x: False)
    
    result = get_user_config()
    assert result == DEFAULT_CONFIG
    
    # Test 6: USER_CONFIG_PATH exists - should load it
    monkeypatch.delenv('COOKIECUTTER_CONFIG', raising=False)
    user_config_file = tmp_path / "user_config.yaml"
    user_config_content = {'abbreviations': {'custom': 'https://custom.com'}}
    user_config_file.write_text(yaml.dump(user_config_content))
    
    def mock_exists(path):
        return str(path) == USER_CONFIG_PATH
    
    monkeypatch.setattr('os.path.exists', mock_exists)
    monkeypatch.setattr('cookiecutter.config.USER_CONFIG_PATH', str(user_config_file))
    
    result = get_user_config()
    assert 'custom' in result['abbreviations']
    
    # Test 7: Invalid config file path should raise ConfigDoesNotExistException
    with pytest.raises(ConfigDoesNotExistException):
        get_user_config(config_file='/nonexistent/path/config.yaml')
    
    # Test 8: COOKIECUTTER_CONFIG points to invalid file should raise error
    monkeypatch.setenv('COOKIECUTTER_CONFIG', '/nonexistent/env/config.yaml')
    monkeypatch.setattr('os.path.exists', lambda x: False)
    
    with pytest.raises(ConfigDoesNotExistException):
        get_user_config()


# LLM-generated content at query #30
#--------------------------

```python
import pytest
import os
import tempfile
from pathlib import Path
import yaml

def test_get_config():
    """Test get_config function with various scenarios."""
    
    # Test 1: Config file does not exist
    with pytest.raises(ConfigDoesNotExistException):
        get_config('/nonexistent/path/config.yaml')
    
    # Test 2: Valid config file with basic content
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        config_content = {
            'cookiecutters_dir': '~/.cookiecutters/',
            'replay_dir': '~/.cookiecutter_replay/',
        }
        yaml.dump(config_content, f)
        temp_config_path = f.name
    
    try:
        result = get_config(temp_config_path)
        assert isinstance(result, dict)
        assert 'cookiecutters_dir' in result
        assert 'replay_dir' in result
        assert result['cookiecutters_dir'] == os.path.expanduser('~/.cookiecutters/')
        assert result['replay_dir'] == os.path.expanduser('~/.cookiecutter_replay/')
    finally:
        os.unlink(temp_config_path)
    
    # Test 3: Config file with environment variables in paths
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        config_content = {
            'cookiecutters_dir': '$HOME/.cookiecutters/',
        }
        yaml.dump(config_content, f)
        temp_config_path = f.name
    
    try:
        result = get_config(temp_config_path)
        assert result['cookiecutters_dir'] == os.path.expandvars('$HOME/.cookiecutters/')
    finally:
        os.unlink(temp_config_path)
    
    # Test 4: Invalid YAML file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write('invalid: yaml: content: [')
        temp_config_path = f.name
    
    try:
        with pytest.raises(InvalidConfiguration):
            get_config(temp_config_path)
    finally:
        os.unlink(temp_config_path)
    
    # Test 5: YAML file with non-dict top-level element
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write('- item1\n- item2\n')
        temp_config_path = f.name
    
    try:
        with pytest.raises(InvalidConfiguration):
            get_config(temp_config_path)
    finally:
        os.unlink(temp_config_path)
    
    # Test 6: Empty YAML file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write('')
        temp_config_path = f.name
    
    try:
        result = get_config(temp_config_path)
        assert isinstance(result, dict)
        assert 'cookiecutters_dir' in result
        assert 'replay_dir' in result
        assert 'default_context' in result
        assert 'abbreviations' in result
    finally:
        os.unlink(temp_config_path)
    
    # Test 7: Config file merges with defaults
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        config_content = {
            'abbreviations': {'custom': 'https://custom.com/{0}'},
        }
        yaml.dump(config_content, f)
        temp_config_path = f.name
    
    try:
        result = get_config(temp_config_path)
        assert 'custom' in result['abbreviations']
        assert 'gh' in result['abbreviations']
        assert result['abbreviations']['gh'] == 'https://github.com/{0}.git'
    finally:
        os.unlink(temp_config_path)
    
    # Test 8: Config with Path object
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        config_content = {'cookiecutters_dir': '/tmp/test'}
        yaml.dump(config_content, f)
        temp_config_path = f.name
    
    try:
        result = get_config(Path(temp_config_path))
        assert isinstance(result, dict)
        assert result['cookiecutters_dir'] == '/tmp/test'
    finally:
        os.unlink(temp_config_path)


# LLM-generated content at query #31
#--------------------------

```python
def test_get_config(tmp_path, monkeypatch):
    """Test get_config function with various scenarios."""
    
    # Test 1: Non-existent config file raises ConfigDoesNotExistException
    with pytest.raises(ConfigDoesNotExistException):
        get_config('/non/existent/path/.cookiecutterrc')
    
    # Test 2: Valid YAML config file is loaded and merged with defaults
    config_file = tmp_path / 'config.yaml'
    config_file.write_text('cookiecutters_dir: /custom/path\n')
    config = get_config(config_file)
    assert config['cookiecutters_dir'] == '/custom/path'
    assert 'replay_dir' in config
    assert 'abbreviations' in config
    
    # Test 3: Empty YAML file uses defaults
    empty_config = tmp_path / 'empty.yaml'
    empty_config.write_text('')
    config = get_config(empty_config)
    assert config == DEFAULT_CONFIG
    
    # Test 4: Invalid YAML raises InvalidConfiguration
    invalid_yaml = tmp_path / 'invalid.yaml'
    invalid_yaml.write_text('invalid: yaml: content: [')
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml)
    
    # Test 5: Non-dict top-level YAML raises InvalidConfiguration
    non_dict_yaml = tmp_path / 'non_dict.yaml'
    non_dict_yaml.write_text('- item1\n- item2\n')
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml)
    
    # Test 6: Path expansion with environment variables
    config_with_env = tmp_path / 'config_env.yaml'
    config_with_env.write_text('cookiecutters_dir: $HOME/.custom_cookiecutters\n')
    config = get_config(config_with_env)
    assert '$HOME' not in config['cookiecutters_dir']
    
    # Test 7: Path expansion with tilde
    config_with_tilde = tmp_path / 'config_tilde.yaml'
    config_with_tilde.write_text('replay_dir: ~/custom_replay\n')
    config = get_config(config_with_tilde)
    assert '~' not in config['replay_dir']
    
    # Test 8: Nested dict merging (abbreviations)
    config_with_abbrev = tmp_path / 'config_abbrev.yaml'
    config_with_abbrev.write_text('abbreviations:\n  custom: https://custom.com/{0}\n')
    config = get_config(config_with_abbrev)
    assert 'gh' in config['abbreviations']
    assert 'custom' in config['abbreviations']
    assert config['abbreviations']['custom'] == 'https://custom.com/{0}'
    
    # Test 9: Default context is preserved
    config_with_context = tmp_path / 'config_context.yaml'
    config_with_context.write_text('default_context:\n  author: John Doe\n')
    config = get_config(config_with_context)
    assert config['default_context']['author'] == 'John Doe'


# LLM-generated content at query #32
#--------------------------

```python
def test_get_config(tmp_path, monkeypatch):
    """Test get_config function with various scenarios."""
    # Test 1: Config file does not exist
    non_existent_path = tmp_path / "non_existent.yml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_path)

    # Test 2: Valid YAML config file
    valid_config_file = tmp_path / "valid_config.yml"
    valid_config_content = """
cookiecutters_dir: /custom/path
replay_dir: /replay/path
default_context:
    author_name: John Doe
abbreviations:
    custom: https://example.com/{0}.git
"""
    valid_config_file.write_text(valid_config_content)
    config = get_config(valid_config_file)
    
    assert config['cookiecutters_dir'].endswith('custom/path')
    assert config['replay_dir'].endswith('replay/path')
    assert config['default_context']['author_name'] == 'John Doe'
    assert config['abbreviations']['custom'] == 'https://example.com/{0}.git'
    assert config['abbreviations']['gh'] == 'https://github.com/{0}.git'

    # Test 3: Invalid YAML file
    invalid_yaml_file = tmp_path / "invalid.yml"
    invalid_yaml_file.write_text("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test 4: YAML file with non-dict top level
    non_dict_yaml_file = tmp_path / "non_dict.yml"
    non_dict_yaml_file.write_text("- item1\n- item2")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)

    # Test 5: Empty YAML file
    empty_yaml_file = tmp_path / "empty.yml"
    empty_yaml_file.write_text("")
    config = get_config(empty_yaml_file)
    assert config == DEFAULT_CONFIG

    # Test 6: Path expansion with environment variables
    env_config_file = tmp_path / "env_config.yml"
    env_config_content = """
cookiecutters_dir: $HOME/.cookiecutters
replay_dir: ~/replay
"""
    env_config_file.write_text(env_config_content)
    config = get_config(env_config_file)
    
    assert '$HOME' not in config['cookiecutters_dir']
    assert '~' not in config['replay_dir']

    # Test 7: Merging with defaults
    partial_config_file = tmp_path / "partial.yml"
    partial_config_content = """
cookiecutters_dir: /custom/cookiecutters
"""
    partial_config_file.write_text(partial_config_content)
    config = get_config(partial_config_file)
    
    assert config['cookiecutters_dir'].endswith('custom/cookiecutters')
    assert 'replay_dir' in config
    assert 'abbreviations' in config
    assert config['abbreviations']['gh'] == 'https://github.com/{0}.git'

    # Test 8: Config file with Path object
    path_config_file = tmp_path / "path_config.yml"
    path_config_file.write_text("cookiecutters_dir: /test/path")
    config = get_config(path_config_file)
    assert config['cookiecutters_dir'].endswith('test/path')


# LLM-generated content at query #33
#--------------------------

```python
def test_get_config(tmp_path):
    """Test get_config function with valid and invalid configurations."""
    # Test with valid YAML config file
    config_file = tmp_path / "cookiecutterrc"
    config_content = """
cookiecutters_dir: ~/.cookiecutters/
replay_dir: ~/.cookiecutter_replay/
default_context:
  full_name: "Test User"
abbreviations:
  gh: https://github.com/{0}.git
"""
    config_file.write_text(config_content, encoding='utf-8')
    
    result = get_config(config_file)
    
    assert isinstance(result, dict)
    assert 'cookiecutters_dir' in result
    assert 'replay_dir' in result
    assert 'default_context' in result
    assert result['default_context']['full_name'] == "Test User"
    assert result['abbreviations']['gh'] == 'https://github.com/{0}.git'
    assert os.path.expanduser('~/.cookiecutters/') == result['cookiecutters_dir']


def test_get_config_file_not_found(tmp_path):
    """Test get_config raises exception when config file does not exist."""
    nonexistent_file = tmp_path / "nonexistent.yaml"
    
    with pytest.raises(ConfigDoesNotExistException):
        get_config(nonexistent_file)


def test_get_config_invalid_yaml(tmp_path):
    """Test get_config raises exception for invalid YAML."""
    config_file = tmp_path / "invalid.yaml"
    config_file.write_text("invalid: yaml: content:", encoding='utf-8')
    
    with pytest.raises(InvalidConfiguration):
        get_config(config_file)


def test_get_config_non_dict_yaml(tmp_path):
    """Test get_config raises exception when YAML root is not a dict."""
    config_file = tmp_path / "list.yaml"
    config_file.write_text("- item1\n- item2\n", encoding='utf-8')
    
    with pytest.raises(InvalidConfiguration):
        get_config(config_file)


def test_get_config_empty_file(tmp_path):
    """Test get_config with empty YAML file returns default config."""
    config_file = tmp_path / "empty.yaml"
    config_file.write_text("", encoding='utf-8')
    
    result = get_config(config_file)
    
    assert isinstance(result, dict)
    assert result['cookiecutters_dir'] == os.path.expanduser('~/.cookiecutters/')
    assert result['replay_dir'] == os.path.expanduser('~/.cookiecutter_replay/')


def test_get_config_with_env_vars(tmp_path):
    """Test get_config expands environment variables in paths."""
    config_file = tmp_path / "config.yaml"
    config_content = """
cookiecutters_dir: $HOME/.cookiecutters/
replay_dir: ~/test_replay/
"""
    config_file.write_text(config_content, encoding='utf-8')
    
    result = get_config(config_file)
    
    assert os.environ.get('HOME') in result['cookiecutters_dir'] or '~' not in result['cookiecutters_dir']
    assert '~' not in result['replay_dir']


def test_get_config_merges_with_defaults(tmp_path):
    """Test get_config merges custom config with defaults."""
    config_file = tmp_path / "partial.yaml"
    config_content = """
default_context:
  custom_key: custom_value
"""
    config_file.write_text(config_content, encoding='utf-8')
    
    result = get_config(config_file)
    
    assert result['default_context']['custom_key'] == 'custom_value'
    assert 'cookiecutters_dir' in result
    assert 'replay_dir' in result
    assert 'abbreviations' in result


def test_get_config_preserves_builtin_abbreviations(tmp_path):
    """Test get_config preserves builtin abbreviations when merging."""
    config_file = tmp_path / "config.yaml"
    config_content = """
abbreviations:
  custom: https://custom.com/{0}
"""
    config_file.write_text(config_content, encoding='utf-8')
    
    result = get_config(config_file)
    
    assert result['abbreviations']['gh'] == 'https://github.com/{0}.git'
    assert result['abbreviations']['gl'] == 'https://gitlab.com/{0}.git'
    assert result['abbreviations']['bb'] == 'https://bitbucket.org/{0}'
    assert result['abbreviations']['custom'] == 'https://custom.com/{0}'


# LLM-generated content at query #34
#--------------------------

```python
def test_get_config(tmp_path, monkeypatch):
    """Test get_config function with various scenarios."""
    # Test 1: Valid YAML config file
    config_file = tmp_path / "cookiecutterrc"
    config_content = """
cookiecutters_dir: /custom/cookiecutters
replay_dir: /custom/replay
default_context:
  author_name: Test Author
abbreviations:
  custom: https://custom.com/{0}.git
"""
    config_file.write_text(config_content)
    
    result = get_config(config_file)
    
    assert result['cookiecutters_dir'] == '/custom/cookiecutters'
    assert result['replay_dir'] == '/custom/replay'
    assert result['default_context']['author_name'] == 'Test Author'
    assert result['abbreviations']['custom'] == 'https://custom.com/{0}.git'
    assert result['abbreviations']['gh'] == 'https://github.com/{0}.git'  # builtin preserved


def test_get_config_nonexistent_file(tmp_path):
    """Test get_config raises exception for non-existent file."""
    nonexistent = tmp_path / "nonexistent.yml"
    
    with pytest.raises(ConfigDoesNotExistException):
        get_config(nonexistent)


def test_get_config_invalid_yaml(tmp_path):
    """Test get_config raises exception for invalid YAML."""
    config_file = tmp_path / "cookiecutterrc"
    config_file.write_text("invalid: yaml: content: [")
    
    with pytest.raises(InvalidConfiguration):
        get_config(config_file)


def test_get_config_non_dict_top_level(tmp_path):
    """Test get_config raises exception when top-level is not a dict."""
    config_file = tmp_path / "cookiecutterrc"
    config_file.write_text("- item1\n- item2")
    
    with pytest.raises(InvalidConfiguration):
        get_config(config_file)


def test_get_config_empty_file(tmp_path):
    """Test get_config with empty YAML file."""
    config_file = tmp_path / "cookiecutterrc"
    config_file.write_text("")
    
    result = get_config(config_file)
    
    assert result == DEFAULT_CONFIG


def test_get_config_path_expansion(tmp_path, monkeypatch):
    """Test get_config expands environment variables and user home."""
    config_file = tmp_path / "cookiecutterrc"
    config_content = """
cookiecutters_dir: $HOME/.cookiecutters_custom
replay_dir: ~/replay_custom
"""
    config_file.write_text(config_content)
    
    monkeypatch.setenv('HOME', str(tmp_path))
    
    result = get_config(config_file)
    
    assert '$HOME' not in result['cookiecutters_dir']
    assert '~' not in result['replay_dir']
    assert result['cookiecutters_dir'].endswith('.cookiecutters_custom')
    assert result['replay_dir'].endswith('replay_custom')


def test_get_config_partial_override(tmp_path):
    """Test get_config merges partial config with defaults."""
    config_file = tmp_path / "cookiecutterrc"
    config_content = """
cookiecutters_dir: /custom/dir
"""
    config_file.write_text(config_content)
    
    result = get_config(config_file)
    
    assert result['cookiecutters_dir'] == '/custom/dir'
    assert 'replay_dir' in result
    assert 'default_context' in result
    assert 'abbreviations' in result


def test_get_config_merge_abbreviations(tmp_path):
    """Test get_config merges abbreviations while preserving builtins."""
    config_file = tmp_path / "cookiecutterrc"
    config_content = """
abbreviations:
  myrepo: https://myrepo.com/{0}.git
"""
    config_file.write_text(config_content)
    
    result = get_config(config_file)
    
    assert result['abbreviations']['myrepo'] == 'https://myrepo.com/{0}.git'
    assert result['abbreviations']['gh'] == 'https://github.com/{0}.git'
    assert result['abbreviations']['gl'] == 'https://gitlab.com/{0}.git'
    assert result['abbreviations']['bb'] == 'https://bitbucket.org/{0}'


# LLM-generated content at query #35
#--------------------------

```python
def test_get_config(tmp_path, monkeypatch):
    """Test get_config function with various scenarios."""
    
    # Test 1: Valid YAML config file
    config_file = tmp_path / "config.yaml"
    config_content = """
cookiecutters_dir: /custom/cookiecutters
replay_dir: /custom/replay
default_context:
    full_name: "Test User"
abbreviations:
    custom: "https://example.com/{0}"
"""
    config_file.write_text(config_content)
    
    result = get_config(config_file)
    assert result['cookiecutters_dir'] == '/custom/cookiecutters'
    assert result['replay_dir'] == '/custom/replay'
    assert result['default_context']['full_name'] == 'Test User'
    assert result['abbreviations']['custom'] == 'https://example.com/{0}'
    # Built-in abbreviations should be preserved
    assert result['abbreviations']['gh'] == 'https://github.com/{0}.git'


def test_get_config_nonexistent_file():
    """Test get_config raises exception for non-existent file."""
    with pytest.raises(ConfigDoesNotExistException):
        get_config('/nonexistent/path/config.yaml')


def test_get_config_invalid_yaml(tmp_path):
    """Test get_config raises exception for invalid YAML."""
    config_file = tmp_path / "invalid.yaml"
    config_file.write_text("invalid: yaml: content: [")
    
    with pytest.raises(InvalidConfiguration):
        get_config(config_file)


def test_get_config_non_dict_top_level(tmp_path):
    """Test get_config raises exception when top-level is not a dict."""
    config_file = tmp_path / "list.yaml"
    config_file.write_text("- item1\n- item2")
    
    with pytest.raises(InvalidConfiguration):
        get_config(config_file)


def test_get_config_empty_file(tmp_path):
    """Test get_config with empty YAML file returns default config."""
    config_file = tmp_path / "empty.yaml"
    config_file.write_text("")
    
    result = get_config(config_file)
    assert 'cookiecutters_dir' in result
    assert 'replay_dir' in result
    assert 'abbreviations' in result


def test_get_config_path_expansion(tmp_path, monkeypatch):
    """Test get_config expands environment variables and home directory."""
    monkeypatch.setenv('TEST_VAR', '/test/path')
    
    config_file = tmp_path / "config.yaml"
    config_content = """
cookiecutters_dir: $TEST_VAR/cookiecutters
replay_dir: ~/replay
"""
    config_file.write_text(config_content)
    
    result = get_config(config_file)
    assert result['cookiecutters_dir'] == '/test/path/cookiecutters'
    assert '~' not in result['replay_dir']
    assert result['replay_dir'].endswith('replay')


def test_get_config_merges_with_defaults(tmp_path):
    """Test get_config merges user config with defaults."""
    config_file = tmp_path / "config.yaml"
    config_content = """
default_context:
    project_name: "My Project"
"""
    config_file.write_text(config_content)
    
    result = get_config(config_file)
    # Should have merged values
    assert result['default_context']['project_name'] == 'My Project'
    # Should preserve default values not overridden
    assert result['abbreviations']['gh'] == 'https://github.com/{0}.git'


def test_get_config_nested_dict_merge(tmp_path):
    """Test get_config properly merges nested dictionaries."""
    config_file = tmp_path / "config.yaml"
    config_content = """
abbreviations:
    custom: "https://custom.com/{0}"
"""
    config_file.write_text(config_content)
    
    result = get_config(config_file)
    # Custom abbreviation should be added
    assert result['abbreviations']['custom'] == 'https://custom.com/{0}'
    # Built-in abbreviations should still exist
    assert 'gh' in result['abbreviations']
    assert 'gl' in result['abbreviations']
    assert 'bb' in result['abbreviations']


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest
import os
import tempfile
import yaml
from pathlib import Path


def test_get_config():
    """Test get_config function with valid YAML configuration file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_file = os.path.join(tmpdir, 'test_config.yaml')
        
        # Create a valid config file
        config_content = {
            'cookiecutters_dir': '~/.cookiecutters/',
            'replay_dir': '~/.cookiecutter_replay/',
            'default_context': {'key': 'value'},
            'abbreviations': {'gh': 'https://github.com/{0}.git'}
        }
        
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config_content, f)
        
        result = get_config(config_file)
        
        assert isinstance(result, dict)
        assert 'cookiecutters_dir' in result
        assert 'replay_dir' in result
        assert 'default_context' in result
        assert 'abbreviations' in result
        assert result['default_context']['key'] == 'value'


def test_get_config_file_not_exists():
    """Test get_config raises ConfigDoesNotExistException when file doesn't exist."""
    with pytest.raises(ConfigDoesNotExistException):
        get_config('/nonexistent/path/config.yaml')


def test_get_config_invalid_yaml():
    """Test get_config raises InvalidConfiguration for invalid YAML."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_file = os.path.join(tmpdir, 'invalid_config.yaml')
        
        # Create an invalid YAML file
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write('invalid: yaml: content: [')
        
        with pytest.raises(InvalidConfiguration):
            get_config(config_file)


def test_get_config_non_dict_yaml():
    """Test get_config raises InvalidConfiguration when top-level is not a dict."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_file = os.path.join(tmpdir, 'list_config.yaml')
        
        # Create a YAML file with list at top level
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(['item1', 'item2'], f)
        
        with pytest.raises(InvalidConfiguration):
            get_config(config_file)


def test_get_config_empty_yaml():
    """Test get_config handles empty YAML file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_file = os.path.join(tmpdir, 'empty_config.yaml')
        
        # Create an empty YAML file
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write('')
        
        result = get_config(config_file)
        
        assert isinstance(result, dict)
        assert result == DEFAULT_CONFIG


def test_get_config_expands_paths():
    """Test get_config expands environment variables and user paths."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_file = os.path.join(tmpdir, 'path_config.yaml')
        
        config_content = {
            'cookiecutters_dir': '~/custom_cookiecutters',
            'replay_dir': '~/.custom_replay'
        }
        
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config_content, f)
        
        result = get_config(config_file)
        
        assert '~' not in result['cookiecutters_dir']
        assert '~' not in result['replay_dir']
        assert result['cookiecutters_dir'].startswith(os.path.expanduser('~'))
        assert result['replay_dir'].startswith(os.path.expanduser('~'))


def test_get_config_merges_with_defaults():
    """Test get_config merges provided config with default values."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_file = os.path.join(tmpdir, 'partial_config.yaml')
        
        config_content = {
            'default_context': {'custom_key': 'custom_value'}
        }
        
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config_content, f)
        
        result = get_config(config_file)
        
        # Should have merged values
        assert 'default_context' in result
        assert 'custom_key' in result['default_context']
        assert result['default_context']['custom_key'] == 'custom_value'
        # Should preserve defaults
        assert 'cookiecutters_dir' in result
        assert 'replay_dir' in result


def test_get_config_with_path_object():
    """Test get_config accepts Path object."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_file = Path(tmpdir) / 'path_obj_config.yaml'
        
        config_content = {'default_context': {}}
        
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config_content, f)
        
        result = get_config(config_file)
        
        assert isinstance(result, dict)
        assert 'default_context' in result


# LLM-generated content at query #2
#--------------------------

```python
def test_get_user_config(tmp_path, monkeypatch):
    """Test get_user_config function with various scenarios."""
    
    # Test 1: default_config=True returns DEFAULT_CONFIG
    result = get_user_config(default_config=True)
    assert result == DEFAULT_CONFIG
    assert result is not DEFAULT_CONFIG  # Should be a copy
    
    # Test 2: default_config as dict merges with DEFAULT_CONFIG
    custom_default = {'cookiecutters_dir': '/custom/path'}
    result = get_user_config(default_config=custom_default)
    assert result['cookiecutters_dir'] == '/custom/path'
    assert result['replay_dir'] == DEFAULT_CONFIG['replay_dir']
    
    # Test 3: No config file exists, no env variable, returns DEFAULT_CONFIG
    monkeypatch.delenv('COOKIECUTTER_CONFIG', raising=False)
    monkeypatch.setattr('os.path.exists', lambda x: False)
    result = get_user_config()
    assert result == DEFAULT_CONFIG
    
    # Test 4: User config file exists at default location
    config_file = tmp_path / '.cookiecutterrc'
    config_content = 'cookiecutters_dir: /tmp/test\nreplay_dir: /tmp/replay'
    config_file.write_text(config_content)
    
    monkeypatch.delenv('COOKIECUTTER_CONFIG', raising=False)
    monkeypatch.setattr('cookiecutter.config.USER_CONFIG_PATH', str(config_file))
    result = get_user_config()
    assert '/tmp/test' in result['cookiecutters_dir']
    
    # Test 5: Custom config_file path is loaded
    custom_config = tmp_path / 'custom.yaml'
    custom_config.write_text('cookiecutters_dir: /custom/test')
    result = get_user_config(config_file=str(custom_config))
    assert '/custom/test' in result['cookiecutters_dir']
    
    # Test 6: COOKIECUTTER_CONFIG environment variable is used
    env_config = tmp_path / 'env_config.yaml'
    env_config.write_text('cookiecutters_dir: /env/path')
    monkeypatch.setenv('COOKIECUTTER_CONFIG', str(env_config))
    monkeypatch.setattr('cookiecutter.config.USER_CONFIG_PATH', '/nonexistent/path')
    result = get_user_config()
    assert '/env/path' in result['cookiecutters_dir']
    
    # Test 7: COOKIECUTTER_CONFIG with nonexistent file raises error
    monkeypatch.setenv('COOKIECUTTER_CONFIG', '/nonexistent/config.yaml')
    with pytest.raises(ConfigDoesNotExistException):
        get_user_config()
    
    # Test 8: Invalid YAML raises error
    invalid_yaml = tmp_path / 'invalid.yaml'
    invalid_yaml.write_text('{ invalid yaml: [')
    with pytest.raises(InvalidConfiguration):
        get_user_config(config_file=str(invalid_yaml))
    
    # Test 9: Non-dict top-level YAML raises error
    non_dict_yaml = tmp_path / 'non_dict.yaml'
    non_dict_yaml.write_text('- item1\n- item2')
    with pytest.raises(InvalidConfiguration):
        get_user_config(config_file=str(non_dict_yaml))
    
    # Test 10: Empty YAML file is treated as empty dict
    empty_yaml = tmp_path / 'empty.yaml'
    empty_yaml.write_text('')
    result = get_user_config(config_file=str(empty_yaml))
    assert result == DEFAULT_CONFIG
    
    # Test 11: config_file=USER_CONFIG_PATH is treated as not specified
    monkeypatch.delenv('COOKIECUTTER_CONFIG', raising=False)
    monkeypatch.setattr('cookiecutter.config.USER_CONFIG_PATH', str(config_file))
    result = get_user_config(config_file=str(config_file))
    assert result == get_user_config()
    
    # Test 12: Paths with environment variables are expanded
    monkeypatch.setenv('TEST_VAR', '/expanded')
    env_var_config = tmp_path / 'env_var.yaml'
    env_var_config.write_text('cookiecutters_dir: $TEST_VAR/cookies')
    result = get_user_config(config_file=str(env_var_config))
    assert '/expanded/cookies' in result['cookiecutters_dir']


# LLM-generated content at query #3
#--------------------------

```python
def test_get_config(tmp_path, monkeypatch):
    """Test get_config function with valid and invalid config files."""
    # Test 1: Config file does not exist
    non_existent_path = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_path)

    # Test 2: Valid config file with basic settings
    config_file = tmp_path / "valid_config.yaml"
    config_content = """
cookiecutters_dir: ~/.cookiecutters/
replay_dir: ~/.cookiecutter_replay/
default_context:
  author_name: Test Author
abbreviations:
  gh: https://github.com/{0}.git
"""
    config_file.write_text(config_content)
    result = get_config(config_file)
    
    assert isinstance(result, dict)
    assert 'cookiecutters_dir' in result
    assert 'replay_dir' in result
    assert 'default_context' in result
    assert 'abbreviations' in result
    assert result['default_context']['author_name'] == 'Test Author'

    # Test 3: Config file with environment variables in paths
    config_file_env = tmp_path / "config_with_env.yaml"
    config_content_env = """
cookiecutters_dir: $HOME/.cookiecutters/
replay_dir: ~/.cookiecutter_replay/
"""
    config_file_env.write_text(config_content_env)
    result = get_config(config_file_env)
    
    assert os.path.expandvars('$HOME') in result['cookiecutters_dir']

    # Test 4: Invalid YAML file
    invalid_yaml_file = tmp_path / "invalid.yaml"
    invalid_yaml_file.write_text("{ invalid yaml content: [")
    
    with pytest.raises(InvalidConfiguration, match="Unable to parse YAML file"):
        get_config(invalid_yaml_file)

    # Test 5: YAML file with non-dict top-level element
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    non_dict_yaml_file.write_text("- item1\n- item2")
    
    with pytest.raises(InvalidConfiguration, match="Top-level element of YAML file.*should be an object"):
        get_config(non_dict_yaml_file)

    # Test 6: Empty YAML file
    empty_yaml_file = tmp_path / "empty.yaml"
    empty_yaml_file.write_text("")
    result = get_config(empty_yaml_file)
    
    assert result == DEFAULT_CONFIG

    # Test 7: Config file merges with defaults
    partial_config_file = tmp_path / "partial_config.yaml"
    partial_config_file.write_text("default_context:\n  custom_key: custom_value")
    result = get_config(partial_config_file)
    
    assert result['default_context']['custom_key'] == 'custom_value'
    assert 'cookiecutters_dir' in result
    assert 'abbreviations' in result

    # Test 8: Paths are expanded correctly
    config_with_tilde = tmp_path / "config_tilde.yaml"
    config_with_tilde.write_text("cookiecutters_dir: ~/my_cookiecutters\nreplay_dir: ~/my_replay")
    result = get_config(config_with_tilde)
    
    assert '~' not in result['cookiecutters_dir']
    assert '~' not in result['replay_dir']
    assert result['cookiecutters_dir'].startswith(os.path.expanduser('~'))
    assert result['replay_dir'].startswith(os.path.expanduser('~'))

    # Test 9: Abbreviations are merged properly
    config_abbrev = tmp_path / "config_abbrev.yaml"
    config_abbrev.write_text("abbreviations:\n  custom: https://custom.com/{0}")
    result = get_config(config_abbrev)
    
    assert 'gh' in result['abbreviations']
    assert 'gl' in result['abbreviations']
    assert 'bb' in result['abbreviations']
    assert result['abbreviations']['custom'] == 'https://custom.com/{0}'


# LLM-generated content at query #4
#--------------------------

```python
def test_get_user_config(tmp_path, monkeypatch):
    """Test get_user_config function with various scenarios."""
    
    # Test 1: Return default config when default_config is True
    result = get_user_config(default_config=True)
    assert result == DEFAULT_CONFIG
    assert isinstance(result, dict)
    assert 'cookiecutters_dir' in result
    assert 'replay_dir' in result
    
    # Test 2: Merge provided dict with defaults when default_config is dict
    custom_dict = {'cookiecutters_dir': '/custom/path'}
    result = get_user_config(default_config=custom_dict)
    assert result['cookiecutters_dir'] == '/custom/path'
    assert result['replay_dir'] == DEFAULT_CONFIG['replay_dir']
    
    # Test 3: Load custom config file
    config_file = tmp_path / 'custom_config.yml'
    config_content = {
        'cookiecutters_dir': '/custom/cookies',
        'replay_dir': '/custom/replay'
    }
    config_file.write_text(yaml.dump(config_content))
    result = get_user_config(config_file=str(config_file))
    assert result['cookiecutters_dir'] == '/custom/cookies'
    assert result['replay_dir'] == '/custom/replay'
    
    # Test 4: Load from environment variable when set
    env_config_file = tmp_path / 'env_config.yml'
    env_config_content = {'default_context': {'key': 'value'}}
    env_config_file.write_text(yaml.dump(env_config_content))
    monkeypatch.setenv('COOKIECUTTER_CONFIG', str(env_config_file))
    monkeypatch.delenv('COOKIECUTTER_CONFIG', raising=False)
    monkeypatch.setenv('COOKIECUTTER_CONFIG', str(env_config_file))
    result = get_user_config()
    assert result['default_context']['key'] == 'value'
    
    # Test 5: Return defaults when no config file exists and no env var set
    monkeypatch.delenv('COOKIECUTTER_CONFIG', raising=False)
    monkeypatch.setattr('os.path.exists', lambda x: False)
    result = get_user_config()
    assert result == DEFAULT_CONFIG
    
    # Test 6: Load from USER_CONFIG_PATH if it exists
    monkeypatch.delenv('COOKIECUTTER_CONFIG', raising=False)
    user_config_file = tmp_path / 'user_config.yml'
    user_config_content = {'abbreviations': {'custom': 'https://custom.com/{0}'}}
    user_config_file.write_text(yaml.dump(user_config_content))
    monkeypatch.setattr('cookiecutter.config.USER_CONFIG_PATH', str(user_config_file))
    result = get_user_config()
    assert result['abbreviations']['custom'] == 'https://custom.com/{0}'
    assert result['abbreviations']['gh'] == BUILTIN_ABBREVIATIONS['gh']
    
    # Test 7: Raise error when environment variable points to non-existent file
    monkeypatch.delenv('COOKIECUTTER_CONFIG', raising=False)
    monkeypatch.setenv('COOKIECUTTER_CONFIG', '/non/existent/path.yml')
    with pytest.raises(ConfigDoesNotExistException):
        get_user_config()
    
    # Test 8: Raise error when custom config file doesn't exist
    with pytest.raises(ConfigDoesNotExistException):
        get_user_config(config_file='/non/existent/config.yml')
    
    # Test 9: Priority - default_config dict takes precedence
    monkeypatch.delenv('COOKIECUTTER_CONFIG', raising=False)
    monkeypatch.setattr('os.path.exists', lambda x: True)
    custom_dict = {'cookiecutters_dir': '/priority/path'}
    result = get_user_config(default_config=custom_dict)
    assert result['cookiecutters_dir'] == '/priority/path'
    
    # Test 10: Invalid YAML raises InvalidConfiguration
    invalid_config_file = tmp_path / 'invalid.yml'
    invalid_config_file.write_text('{ invalid yaml content: [')
    with pytest.raises(InvalidConfiguration):
        get_user_config(config_file=str(invalid_config_file))
    
    # Test 11: Non-dict YAML raises InvalidConfiguration
    non_dict_config_file = tmp_path / 'non_dict.yml'
    non_dict_config_file.write_text('- item1\n- item2')
    with pytest.raises(InvalidConfiguration):
        get_user_config(config_file=str(non_dict_config_file))


# LLM-generated content at query #5
#--------------------------

```python
def test_get_config(tmp_path, monkeypatch):
    """Test get_config function with various scenarios."""
    
    # Test 1: Valid YAML config file
    config_file = tmp_path / "valid_config.yaml"
    config_file.write_text(
        "cookiecutters_dir: ~/my_cookiecutters\n"
        "replay_dir: ~/my_replay\n"
        "default_context:\n"
        "  author_name: John Doe\n"
        "abbreviations:\n"
        "  custom: https://example.com/{0}.git\n"
    )
    
    result = get_config(config_file)
    assert result['cookiecutters_dir'].endswith('my_cookiecutters')
    assert result['replay_dir'].endswith('my_replay')
    assert result['default_context']['author_name'] == 'John Doe'
    assert result['abbreviations']['gh'] == 'https://github.com/{0}.git'
    assert result['abbreviations']['custom'] == 'https://example.com/{0}.git'
    
    # Test 2: Config file with environment variables
    config_file2 = tmp_path / "env_config.yaml"
    config_file2.write_text(
        "cookiecutters_dir: $HOME/.cookiecutters\n"
    )
    
    result2 = get_config(config_file2)
    assert '.cookiecutters' in result2['cookiecutters_dir']
    
    # Test 3: Empty YAML file (should use defaults)
    config_file3 = tmp_path / "empty_config.yaml"
    config_file3.write_text("")
    
    result3 = get_config(config_file3)
    assert 'cookiecutters_dir' in result3
    assert 'replay_dir' in result3
    assert result3['abbreviations']['gh'] == 'https://github.com/{0}.git'
    
    # Test 4: Config file does not exist
    nonexistent_file = tmp_path / "nonexistent.yaml"
    
    with pytest.raises(ConfigDoesNotExistException):
        get_config(nonexistent_file)
    
    # Test 5: Invalid YAML syntax
    invalid_yaml_file = tmp_path / "invalid.yaml"
    invalid_yaml_file.write_text(
        "cookiecutters_dir: ~/my_cookiecutters\n"
        "  invalid: : : syntax\n"
    )
    
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)
    
    # Test 6: YAML file with non-dict top-level element
    non_dict_file = tmp_path / "non_dict.yaml"
    non_dict_file.write_text("- item1\n- item2\n")
    
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_file)
    
    # Test 7: Partial config (only some keys provided)
    partial_config_file = tmp_path / "partial_config.yaml"
    partial_config_file.write_text(
        "default_context:\n"
        "  project_name: My Project\n"
    )
    
    result7 = get_config(partial_config_file)
    assert result7['default_context']['project_name'] == 'My Project'
    assert 'cookiecutters_dir' in result7
    assert 'replay_dir' in result7


# LLM-generated content at query #6
#--------------------------

```python
def test_get_config(tmp_path, monkeypatch):
    """Test get_config function with valid and invalid configurations."""
    # Test 1: Valid config file
    config_file = tmp_path / "valid_config.yaml"
    config_content = """
cookiecutters_dir: /tmp/cookiecutters
replay_dir: /tmp/replay
default_context:
    author_name: John Doe
abbreviations:
    gh: https://github.com/{0}.git
"""
    config_file.write_text(config_content)
    
    result = get_config(config_file)
    
    assert isinstance(result, dict)
    assert 'cookiecutters_dir' in result
    assert 'replay_dir' in result
    assert 'default_context' in result
    assert 'abbreviations' in result
    assert result['default_context']['author_name'] == 'John Doe'

    # Test 2: Config file with environment variable expansion
    config_file_with_env = tmp_path / "config_with_env.yaml"
    config_file_with_env.write_text("cookiecutters_dir: $HOME/.cookiecutters")
    
    result = get_config(config_file_with_env)
    assert '$HOME' not in result['cookiecutters_dir']
    assert result['cookiecutters_dir'].startswith(os.path.expandvars('$HOME'))

    # Test 3: Config file with tilde expansion
    config_file_with_tilde = tmp_path / "config_with_tilde.yaml"
    config_file_with_tilde.write_text("replay_dir: ~/replay")
    
    result = get_config(config_file_with_tilde)
    assert '~' not in result['replay_dir']
    assert result['replay_dir'].startswith(os.path.expanduser('~'))

    # Test 4: Empty config file
    empty_config = tmp_path / "empty_config.yaml"
    empty_config.write_text("")
    
    result = get_config(empty_config)
    assert result == DEFAULT_CONFIG

    # Test 5: Config file does not exist
    non_existent_file = tmp_path / "non_existent.yaml"
    
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)

    # Test 6: Invalid YAML
    invalid_yaml_file = tmp_path / "invalid.yaml"
    invalid_yaml_file.write_text("{ invalid yaml content: [")
    
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test 7: YAML with non-dict top-level element
    non_dict_yaml = tmp_path / "non_dict.yaml"
    non_dict_yaml.write_text("- item1\n- item2")
    
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml)

    # Test 8: Config merges with defaults
    partial_config = tmp_path / "partial_config.yaml"
    partial_config.write_text("cookiecutters_dir: /custom/path")
    
    result = get_config(partial_config)
    assert result['cookiecutters_dir'] == '/custom/path'
    assert 'replay_dir' in result
    assert 'default_context' in result
    assert 'abbreviations' in result

    # Test 9: Config with nested dict merging
    nested_config = tmp_path / "nested_config.yaml"
    nested_config.write_text("""
abbreviations:
    custom: https://custom.com/{0}
""")
    
    result = get_config(nested_config)
    assert 'gh' in result['abbreviations']
    assert 'custom' in result['abbreviations']
    assert result['abbreviations']['custom'] == 'https://custom.com/{0}'


# LLM-generated content at query #7
#--------------------------

```python
def test_get_config(tmp_path, monkeypatch):
    """Test get_config function with various scenarios."""
    
    # Test 1: Non-existent config file raises ConfigDoesNotExistException
    with pytest.raises(ConfigDoesNotExistException):
        get_config('/non/existent/path/.cookiecutterrc')
    
    # Test 2: Valid YAML config file is loaded and merged with defaults
    config_file = tmp_path / 'test_config.yml'
    config_content = {
        'cookiecutters_dir': '~/.custom_cookiecutters/',
        'replay_dir': '~/.custom_replay/',
        'default_context': {'author': 'Test Author'},
    }
    config_file.write_text(yaml.dump(config_content))
    
    result = get_config(config_file)
    
    assert result['cookiecutters_dir'] == os.path.expanduser('~/.custom_cookiecutters/')
    assert result['replay_dir'] == os.path.expanduser('~/.custom_replay/')
    assert result['default_context']['author'] == 'Test Author'
    assert result['abbreviations'] == BUILTIN_ABBREVIATIONS
    
    # Test 3: Empty YAML file returns default config
    empty_config_file = tmp_path / 'empty_config.yml'
    empty_config_file.write_text('')
    
    result = get_config(empty_config_file)
    
    assert result == DEFAULT_CONFIG
    
    # Test 4: Invalid YAML raises InvalidConfiguration
    invalid_config_file = tmp_path / 'invalid_config.yml'
    invalid_config_file.write_text('{ invalid: yaml: content: [')
    
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_config_file)
    
    # Test 5: Non-dict YAML raises InvalidConfiguration
    non_dict_config_file = tmp_path / 'non_dict_config.yml'
    non_dict_config_file.write_text('- item1\n- item2')
    
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_config_file)
    
    # Test 6: Environment variables in paths are expanded
    config_with_env = tmp_path / 'env_config.yml'
    monkeypatch.setenv('TEST_DIR', '/test/directory')
    config_content = {
        'cookiecutters_dir': '$TEST_DIR/cookiecutters/',
        'replay_dir': '$TEST_DIR/replay/',
    }
    config_with_env.write_text(yaml.dump(config_content))
    
    result = get_config(config_with_env)
    
    assert result['cookiecutters_dir'] == '/test/directory/cookiecutters/'
    assert result['replay_dir'] == '/test/directory/replay/'
    
    # Test 7: Custom abbreviations are merged with defaults
    custom_abbrev_file = tmp_path / 'abbrev_config.yml'
    config_content = {
        'abbreviations': {'custom': 'https://custom.com/{0}.git'}
    }
    custom_abbrev_file.write_text(yaml.dump(config_content))
    
    result = get_config(custom_abbrev_file)
    
    assert 'gh' in result['abbreviations']
    assert 'custom' in result['abbreviations']
    assert result['abbreviations']['custom'] == 'https://custom.com/{0}.git'


# LLM-generated content at query #8
#--------------------------

```python
def test_get_user_config(tmp_path, monkeypatch):
    """Test get_user_config function with various scenarios."""
    
    # Test 1: Return default config when default_config=True
    result = get_user_config(default_config=True)
    assert result == DEFAULT_CONFIG
    assert isinstance(result, dict)
    assert 'cookiecutters_dir' in result
    assert 'replay_dir' in result
    
    # Test 2: Merge custom dict with defaults when default_config is a dict
    custom_config = {'cookiecutters_dir': '/custom/path'}
    result = get_user_config(default_config=custom_config)
    assert result['cookiecutters_dir'] == '/custom/path'
    assert result['replay_dir'] == DEFAULT_CONFIG['replay_dir']
    
    # Test 3: Load custom config file
    config_file = tmp_path / "custom_config.yml"
    config_file.write_text("cookiecutters_dir: /tmp/custom\nabbreviations:\n  gh: 'https://github.com/{0}.git'")
    result = get_user_config(config_file=str(config_file))
    assert result['cookiecutters_dir'] == '/tmp/custom'
    
    # Test 4: Load from COOKIECUTTER_CONFIG environment variable
    env_config_file = tmp_path / "env_config.yml"
    env_config_file.write_text("cookiecutters_dir: /env/path")
    monkeypatch.setenv('COOKIECUTTER_CONFIG', str(env_config_file))
    monkeypatch.delenv('COOKIECUTTER_CONFIG', raising=False)
    monkeypatch.setenv('COOKIECUTTER_CONFIG', str(env_config_file))
    result = get_user_config()
    assert result['cookiecutters_dir'] == '/env/path'
    
    # Test 5: Return defaults when no config file exists and no env variable
    monkeypatch.delenv('COOKIECUTTER_CONFIG', raising=False)
    monkeypatch.setattr('cookiecutter.config.USER_CONFIG_PATH', '/nonexistent/path')
    result = get_user_config()
    assert result == DEFAULT_CONFIG
    
    # Test 6: Load user config from default location if it exists
    user_config_path = tmp_path / ".cookiecutterrc"
    user_config_path.write_text("cookiecutters_dir: /user/path")
    monkeypatch.delenv('COOKIECUTTER_CONFIG', raising=False)
    monkeypatch.setattr('cookiecutter.config.USER_CONFIG_PATH', str(user_config_path))
    result = get_user_config()
    assert result['cookiecutters_dir'] == '/user/path'
    
    # Test 7: Raise error for non-existent config file from environment variable
    monkeypatch.delenv('COOKIECUTTER_CONFIG', raising=False)
    monkeypatch.setenv('COOKIECUTTER_CONFIG', '/nonexistent/env/config.yml')
    with pytest.raises(ConfigDoesNotExistException):
        get_user_config()
    
    # Test 8: default_config=True takes precedence over config_file
    result = get_user_config(config_file=str(config_file), default_config=True)
    assert result == DEFAULT_CONFIG
    
    # Test 9: default_config dict takes precedence over config_file
    custom_dict = {'replay_dir': '/custom/replay'}
    result = get_user_config(config_file=str(config_file), default_config=custom_dict)
    assert result['replay_dir'] == '/custom/replay'
    assert 'cookiecutters_dir' in result


# LLM-generated content at query #9
#--------------------------

```python
def test_get_config(tmp_path, monkeypatch):
    """Test get_config function with various scenarios."""
    
    # Test 1: Config file does not exist
    with pytest.raises(ConfigDoesNotExistException):
        get_config('/nonexistent/path/config.yaml')
    
    # Test 2: Valid config file with basic content
    config_file = tmp_path / "config.yaml"
    config_file.write_text("cookiecutters_dir: /custom/path\n")
    result = get_config(config_file)
    assert result['cookiecutters_dir'] == '/custom/path'
    assert 'replay_dir' in result
    assert 'default_context' in result
    
    # Test 3: Config file with environment variable expansion
    config_file = tmp_path / "config_env.yaml"
    config_file.write_text("cookiecutters_dir: $HOME/.custom_cookies\n")
    monkeypatch.setenv('HOME', '/home/testuser')
    result = get_config(config_file)
    assert result['cookiecutters_dir'] == '/home/testuser/.custom_cookies'
    
    # Test 4: Config file with tilde expansion
    config_file = tmp_path / "config_tilde.yaml"
    config_file.write_text("replay_dir: ~/.custom_replay\n")
    result = get_config(config_file)
    assert '~' not in result['replay_dir']
    
    # Test 5: Invalid YAML syntax
    config_file = tmp_path / "config_invalid.yaml"
    config_file.write_text("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(config_file)
    
    # Test 6: YAML file with non-dict top-level element
    config_file = tmp_path / "config_list.yaml"
    config_file.write_text("- item1\n- item2\n")
    with pytest.raises(InvalidConfiguration):
        get_config(config_file)
    
    # Test 7: Empty YAML file
    config_file = tmp_path / "config_empty.yaml"
    config_file.write_text("")
    result = get_config(config_file)
    assert result == DEFAULT_CONFIG
    
    # Test 8: Config with abbreviations merged
    config_file = tmp_path / "config_abbrev.yaml"
    config_file.write_text("abbreviations:\n  custom: 'https://custom.com/{0}.git'\n")
    result = get_config(config_file)
    assert 'gh' in result['abbreviations']
    assert 'custom' in result['abbreviations']
    assert result['abbreviations']['gh'] == 'https://github.com/{0}.git'
    assert result['abbreviations']['custom'] == 'https://custom.com/{0}.git'
    
    # Test 9: Config with default_context
    config_file = tmp_path / "config_context.yaml"
    config_file.write_text("default_context:\n  project_name: my_project\n  author: John Doe\n")
    result = get_config(config_file)
    assert result['default_context']['project_name'] == 'my_project'
    assert result['default_context']['author'] == 'John Doe'
    
    # Test 10: Config with string path instead of Path object
    config_file = tmp_path / "config_str.yaml"
    config_file.write_text("cookiecutters_dir: /path/to/cookies\n")
    result = get_config(str(config_file))
    assert result['cookiecutters_dir'] == '/path/to/cookies'


# LLM-generated content at query #10
#--------------------------

```python
def test_get_config(tmp_path, monkeypatch):
    """Test get_config function with various scenarios."""
    # Test 1: Config file does not exist
    non_existent_path = tmp_path / "non_existent.yml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_path)

    # Test 2: Valid YAML config file
    config_file = tmp_path / "valid_config.yml"
    config_content = {
        'cookiecutters_dir': '/tmp/cookies',
        'replay_dir': '/tmp/replay',
        'default_context': {'project_name': 'my_project'},
    }
    config_file.write_text(yaml.dump(config_content))
    
    result = get_config(config_file)
    assert result['cookiecutters_dir'] == '/tmp/cookies'
    assert result['replay_dir'] == '/tmp/replay'
    assert result['default_context']['project_name'] == 'my_project'
    assert 'abbreviations' in result

    # Test 3: Empty YAML file (should return defaults)
    empty_config_file = tmp_path / "empty_config.yml"
    empty_config_file.write_text("")
    
    result = get_config(empty_config_file)
    assert result == DEFAULT_CONFIG

    # Test 4: Invalid YAML syntax
    invalid_yaml_file = tmp_path / "invalid.yml"
    invalid_yaml_file.write_text("{ invalid yaml: [")
    
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test 5: YAML file with non-dict top-level element
    non_dict_yaml_file = tmp_path / "non_dict.yml"
    non_dict_yaml_file.write_text("- item1\n- item2")
    
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)

    # Test 6: Path expansion with environment variables
    config_with_env_vars = tmp_path / "env_config.yml"
    monkeypatch.setenv('TEST_DIR', str(tmp_path))
    config_content = {
        'cookiecutters_dir': '$TEST_DIR/cookies',
        'replay_dir': '~/replay',
    }
    config_with_env_vars.write_text(yaml.dump(config_content))
    
    result = get_config(config_with_env_vars)
    assert str(tmp_path) in result['cookiecutters_dir']
    assert result['replay_dir'].startswith(os.path.expanduser('~'))

    # Test 7: Merging with defaults preserves abbreviations
    partial_config_file = tmp_path / "partial_config.yml"
    partial_config = {
        'abbreviations': {'custom': 'https://custom.com/{0}.git'},
    }
    partial_config_file.write_text(yaml.dump(partial_config))
    
    result = get_config(partial_config_file)
    assert 'custom' in result['abbreviations']
    assert 'gh' in result['abbreviations']
    assert result['abbreviations']['gh'] == BUILTIN_ABBREVIATIONS['gh']

    # Test 8: Config with all custom values
    full_config_file = tmp_path / "full_config.yml"
    full_config = {
        'cookiecutters_dir': '/custom/cookies',
        'replay_dir': '/custom/replay',
        'default_context': {
            'author_name': 'John Doe',
            'project_slug': 'my_project',
        },
        'abbreviations': {
            'gh': 'https://github.company.com/{0}.git',
        },
    }
    full_config_file.write_text(yaml.dump(full_config))
    
    result = get_config(full_config_file)
    assert result['cookiecutters_dir'] == '/custom/cookies'
    assert result['replay_dir'] == '/custom/replay'
    assert result['default_context']['author_name'] == 'John Doe'
    assert result['abbreviations']['gh'] == 'https://github.company.com/{0}.git'


# LLM-generated content at query #11
#--------------------------

```python
def test_get_config(tmp_path, monkeypatch):
    """Test get_config function with various scenarios."""
    
    # Test 1: Config file does not exist
    with pytest.raises(ConfigDoesNotExistException):
        get_config('/nonexistent/path/config.yaml')
    
    # Test 2: Valid config file with all parameters
    config_file = tmp_path / 'config.yaml'
    config_content = """
cookiecutters_dir: ~/my_cookiecutters/
replay_dir: ~/my_replay/
default_context:
  author_name: John Doe
abbreviations:
  custom: https://custom.com/{0}.git
"""
    config_file.write_text(config_content)
    result = get_config(config_file)
    
    assert 'cookiecutters_dir' in result
    assert 'replay_dir' in result
    assert result['default_context']['author_name'] == 'John Doe'
    assert result['abbreviations']['custom'] == 'https://custom.com/{0}.git'
    assert result['abbreviations']['gh'] == 'https://github.com/{0}.git'
    
    # Test 3: Empty config file
    empty_config = tmp_path / 'empty.yaml'
    empty_config.write_text('')
    result = get_config(empty_config)
    
    assert result == DEFAULT_CONFIG
    
    # Test 4: Config with only some parameters
    partial_config = tmp_path / 'partial.yaml'
    partial_content = """
default_context:
  project_name: my_project
"""
    partial_config.write_text(partial_content)
    result = get_config(partial_config)
    
    assert result['default_context']['project_name'] == 'my_project'
    assert 'cookiecutters_dir' in result
    
    # Test 5: Invalid YAML syntax
    invalid_yaml = tmp_path / 'invalid.yaml'
    invalid_yaml.write_text('invalid: yaml: content:')
    
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml)
    
    # Test 6: YAML file with non-dict top-level element
    non_dict_yaml = tmp_path / 'nondict.yaml'
    non_dict_yaml.write_text('- item1\n- item2')
    
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml)
    
    # Test 7: Path expansion with environment variables
    env_config = tmp_path / 'env.yaml'
    monkeypatch.setenv('TEST_DIR', str(tmp_path))
    env_content = """
cookiecutters_dir: $TEST_DIR/cookies/
replay_dir: ~/replay/
"""
    env_config.write_text(env_content)
    result = get_config(env_config)
    
    assert str(tmp_path) in result['cookiecutters_dir']
    assert '~' not in result['replay_dir']
    
    # Test 8: Config with abbreviations merging
    abbrev_config = tmp_path / 'abbrev.yaml'
    abbrev_content = """
abbreviations:
  gh: https://github.custom.com/{0}.git
  new: https://newsite.com/{0}.git
"""
    abbrev_config.write_text(abbrev_content)
    result = get_config(abbrev_config)
    
    assert result['abbreviations']['gh'] == 'https://github.custom.com/{0}.git'
    assert result['abbreviations']['new'] == 'https://newsite.com/{0}.git'
    assert result['abbreviations']['gl'] == 'https://gitlab.com/{0}.git'


# LLM-generated content at query #12
#--------------------------

```python
def test_get_config(tmp_path, monkeypatch):
    """Test get_config function with various scenarios."""
    # Test 1: Config file does not exist
    with pytest.raises(ConfigDoesNotExistException):
        get_config(tmp_path / "nonexistent.yaml")

    # Test 2: Valid config file with basic content
    config_file = tmp_path / "valid_config.yaml"
    config_file.write_text("cookiecutters_dir: ~/custom_cookiecutters/\n")
    result = get_config(config_file)
    assert result['cookiecutters_dir'].endswith('custom_cookiecutters/')
    assert 'replay_dir' in result
    assert 'abbreviations' in result

    # Test 3: Config file with environment variable expansion
    config_file = tmp_path / "env_config.yaml"
    monkeypatch.setenv('TEST_VAR', '/test/path')
    config_file.write_text("cookiecutters_dir: $TEST_VAR/cookiecutters\n")
    result = get_config(config_file)
    assert '/test/path/cookiecutters' in result['cookiecutters_dir']

    # Test 4: Config file with user home expansion
    config_file = tmp_path / "home_config.yaml"
    config_file.write_text("replay_dir: ~/custom_replay/\n")
    result = get_config(config_file)
    assert '~' not in result['replay_dir']
    assert result['replay_dir'].endswith('custom_replay/')

    # Test 5: Invalid YAML content
    invalid_yaml_file = tmp_path / "invalid.yaml"
    invalid_yaml_file.write_text("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test 6: YAML file with non-dict top-level element
    non_dict_file = tmp_path / "non_dict.yaml"
    non_dict_file.write_text("- item1\n- item2\n")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_file)

    # Test 7: Empty YAML file (should use defaults)
    empty_file = tmp_path / "empty.yaml"
    empty_file.write_text("")
    result = get_config(empty_file)
    assert result['cookiecutters_dir'] == DEFAULT_CONFIG['cookiecutters_dir']
    assert result['replay_dir'] == DEFAULT_CONFIG['replay_dir']

    # Test 8: Config with custom abbreviations merged with defaults
    abbrev_file = tmp_path / "abbrev_config.yaml"
    abbrev_file.write_text("abbreviations:\n  custom: 'https://custom.com/{0}'\n")
    result = get_config(abbrev_file)
    assert 'gh' in result['abbreviations']  # Default preserved
    assert 'custom' in result['abbreviations']  # Custom added
    assert result['abbreviations']['custom'] == 'https://custom.com/{0}'

    # Test 9: Config with default_context
    context_file = tmp_path / "context_config.yaml"
    context_file.write_text("default_context:\n  author: John Doe\n  email: john@example.com\n")
    result = get_config(context_file)
    assert result['default_context']['author'] == 'John Doe'
    assert result['default_context']['email'] == 'john@example.com'

    # Test 10: Config file path as string
    str_config_file = str(tmp_path / "string_path_config.yaml")
    Path(str_config_file).write_text("cookiecutters_dir: ~/test/\n")
    result = get_config(str_config_file)
    assert 'cookiecutters_dir' in result


# LLM-generated content at query #13
#--------------------------

```python
def test_get_config(tmp_path):
    """Test get_config function with valid and invalid configurations."""
    # Test with valid YAML config file
    config_file = tmp_path / "config.yaml"
    config_content = """
cookiecutters_dir: /tmp/cookiecutters
replay_dir: /tmp/replay
default_context:
  project_name: my_project
abbreviations:
  gh: https://github.com/{0}.git
"""
    config_file.write_text(config_content)
    
    result = get_config(config_file)
    
    assert isinstance(result, dict)
    assert 'cookiecutters_dir' in result
    assert 'replay_dir' in result
    assert 'default_context' in result
    assert 'abbreviations' in result
    assert result['default_context']['project_name'] == 'my_project'


def test_get_config_with_environment_variables(tmp_path):
    """Test get_config expands environment variables in paths."""
    config_file = tmp_path / "config.yaml"
    config_content = """
cookiecutters_dir: $HOME/.cookiecutters
replay_dir: ~/replay
"""
    config_file.write_text(config_content)
    
    result = get_config(config_file)
    
    assert '$HOME' not in result['cookiecutters_dir']
    assert '~' not in result['replay_dir']
    assert result['cookiecutters_dir'].startswith('/')
    assert result['replay_dir'].startswith('/')


def test_get_config_file_does_not_exist():
    """Test get_config raises exception when file does not exist."""
    with pytest.raises(ConfigDoesNotExistException):
        get_config('/nonexistent/path/config.yaml')


def test_get_config_invalid_yaml(tmp_path):
    """Test get_config raises exception for invalid YAML."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("invalid: yaml: content:")
    
    with pytest.raises(InvalidConfiguration):
        get_config(config_file)


def test_get_config_non_dict_yaml(tmp_path):
    """Test get_config raises exception when YAML is not a dict."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("- item1\n- item2")
    
    with pytest.raises(InvalidConfiguration):
        get_config(config_file)


def test_get_config_merges_with_defaults(tmp_path):
    """Test get_config merges user config with default config."""
    config_file = tmp_path / "config.yaml"
    config_content = """
default_context:
  custom_key: custom_value
"""
    config_file.write_text(config_content)
    
    result = get_config(config_file)
    
    # Should have defaults
    assert 'cookiecutters_dir' in result
    assert 'replay_dir' in result
    assert 'abbreviations' in result
    # Should have custom values
    assert result['default_context']['custom_key'] == 'custom_value'
    # Should preserve builtin abbreviations
    assert result['abbreviations']['gh'] == 'https://github.com/{0}.git'


def test_get_config_empty_yaml(tmp_path):
    """Test get_config with empty YAML file."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("")
    
    result = get_config(config_file)
    
    assert isinstance(result, dict)
    assert result == DEFAULT_CONFIG


def test_get_config_preserves_nested_dict_values(tmp_path):
    """Test get_config preserves nested dictionary values."""
    config_file = tmp_path / "config.yaml"
    config_content = """
abbreviations:
  custom: https://custom.com/{0}
"""
    config_file.write_text(config_content)
    
    result = get_config(config_file)
    
    # Should preserve builtin abbreviations
    assert 'gh' in result['abbreviations']
    assert 'gl' in result['abbreviations']
    assert 'bb' in result['abbreviations']
    # Should add custom abbreviation
    assert result['abbreviations']['custom'] == 'https://custom.com/{0}'


# LLM-generated content at query #14
#--------------------------

```python
def test_get_config(tmp_path, monkeypatch):
    """Test get_config function."""
    # Test: ConfigDoesNotExistException when file doesn't exist
    with pytest.raises(ConfigDoesNotExistException):
        get_config('/nonexistent/path/config.yaml')

    # Test: Valid YAML config file
    config_file = tmp_path / 'config.yaml'
    config_file.write_text(
        'cookiecutters_dir: ~/my_cookiecutters\n'
        'replay_dir: ~/my_replay\n'
        'abbreviations:\n'
        '  my_abbr: https://example.com/{0}.git\n'
    )
    result = get_config(config_file)
    assert 'cookiecutters_dir' in result
    assert 'replay_dir' in result
    assert result['cookiecutters_dir'] == os.path.expanduser('~/my_cookiecutters')
    assert result['replay_dir'] == os.path.expanduser('~/my_replay')
    assert 'my_abbr' in result['abbreviations']
    assert result['abbreviations']['gh'] == 'https://github.com/{0}.git'

    # Test: Invalid YAML syntax
    invalid_yaml_file = tmp_path / 'invalid.yaml'
    invalid_yaml_file.write_text('invalid: yaml: content: [')
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test: YAML file with non-dict top-level element
    non_dict_file = tmp_path / 'non_dict.yaml'
    non_dict_file.write_text('- item1\n- item2\n')
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_file)

    # Test: Empty YAML file
    empty_file = tmp_path / 'empty.yaml'
    empty_file.write_text('')
    result = get_config(empty_file)
    assert result == DEFAULT_CONFIG

    # Test: Environment variable expansion in paths
    config_with_env = tmp_path / 'config_env.yaml'
    config_with_env.write_text(
        'cookiecutters_dir: $HOME/.cookiecutters\n'
        'replay_dir: $HOME/.replay\n'
    )
    result = get_config(config_with_env)
    assert result['cookiecutters_dir'] == os.path.expandvars(os.path.expanduser('$HOME/.cookiecutters'))
    assert result['replay_dir'] == os.path.expandvars(os.path.expanduser('$HOME/.replay'))

    # Test: Merging with defaults
    partial_config_file = tmp_path / 'partial.yaml'
    partial_config_file.write_text('default_context:\n  project_name: my_project\n')
    result = get_config(partial_config_file)
    assert 'cookiecutters_dir' in result
    assert 'abbreviations' in result
    assert result['abbreviations']['gh'] == 'https://github.com/{0}.git'


# LLM-generated content at query #15
#--------------------------

```python
def test_get_config(tmp_path):
    """Test get_config function with various scenarios."""
    
    # Test 1: Valid config file
    config_file = tmp_path / "config.yaml"
    config_content = """
cookiecutters_dir: /custom/cookiecutters
replay_dir: /custom/replay
default_context:
  author_name: John Doe
abbreviations:
  custom: https://custom.com/{0}
"""
    config_file.write_text(config_content)
    
    result = get_config(config_file)
    assert result['cookiecutters_dir'] == '/custom/cookiecutters'
    assert result['replay_dir'] == '/custom/replay'
    assert result['default_context']['author_name'] == 'John Doe'
    assert result['abbreviations']['custom'] == 'https://custom.com/{0}'
    # Builtin abbreviations should be preserved
    assert result['abbreviations']['gh'] == 'https://github.com/{0}.git'
    
    # Test 2: Config file does not exist
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_file)
    
    # Test 3: Invalid YAML file
    invalid_yaml_file = tmp_path / "invalid.yaml"
    invalid_yaml_file.write_text("{ invalid yaml content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)
    
    # Test 4: YAML file with non-dict top-level element
    non_dict_yaml_file = tmp_path / "non_dict.yaml"
    non_dict_yaml_file.write_text("- item1\n- item2")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml_file)
    
    # Test 5: Empty YAML file
    empty_yaml_file = tmp_path / "empty.yaml"
    empty_yaml_file.write_text("")
    result = get_config(empty_yaml_file)
    assert result == DEFAULT_CONFIG
    
    # Test 6: Path expansion with environment variables
    config_with_env_vars = tmp_path / "config_env.yaml"
    config_with_env_vars.write_text("cookiecutters_dir: $HOME/.cookiecutters")
    result = get_config(config_with_env_vars)
    assert result['cookiecutters_dir'] == os.path.expanduser('~/.cookiecutters')
    
    # Test 7: Path expansion with tilde
    config_with_tilde = tmp_path / "config_tilde.yaml"
    config_with_tilde.write_text("replay_dir: ~/my_replay")
    result = get_config(config_with_tilde)
    assert result['replay_dir'] == os.path.expanduser('~/my_replay')
    
    # Test 8: Partial config merges with defaults
    partial_config_file = tmp_path / "partial.yaml"
    partial_config_file.write_text("default_context:\n  key: value")
    result = get_config(partial_config_file)
    assert 'cookiecutters_dir' in result
    assert 'replay_dir' in result
    assert 'abbreviations' in result
    assert result['default_context']['key'] == 'value'


# LLM-generated content at query #16
#--------------------------

```python
def test_get_config(tmp_path, monkeypatch):
    """Test get_config function with various scenarios."""
    
    # Test 1: Non-existent config file raises ConfigDoesNotExistException
    with pytest.raises(ConfigDoesNotExistException):
        get_config('/nonexistent/path/config.yaml')
    
    # Test 2: Valid YAML config file is loaded and merged with defaults
    config_file = tmp_path / 'config.yaml'
    config_file.write_text('cookiecutters_dir: /custom/path\n')
    config = get_config(config_file)
    assert config['cookiecutters_dir'] == '/custom/path'
    assert 'replay_dir' in config
    assert config['abbreviations'] == BUILTIN_ABBREVIATIONS
    
    # Test 3: Empty YAML file returns default config
    empty_config = tmp_path / 'empty.yaml'
    empty_config.write_text('')
    config = get_config(empty_config)
    assert config == DEFAULT_CONFIG
    
    # Test 4: Invalid YAML raises InvalidConfiguration
    invalid_yaml = tmp_path / 'invalid.yaml'
    invalid_yaml.write_text('invalid: yaml: content: [')
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml)
    
    # Test 5: Non-dict top-level element raises InvalidConfiguration
    non_dict_yaml = tmp_path / 'non_dict.yaml'
    non_dict_yaml.write_text('- item1\n- item2\n')
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml)
    
    # Test 6: Path expansion with environment variables
    config_with_env = tmp_path / 'config_env.yaml'
    config_with_env.write_text('cookiecutters_dir: $HOME/.custom\n')
    monkeypatch.setenv('HOME', '/home/user')
    config = get_config(config_with_env)
    assert config['cookiecutters_dir'] == '/home/user/.custom'
    
    # Test 7: Path expansion with tilde
    config_with_tilde = tmp_path / 'config_tilde.yaml'
    config_with_tilde.write_text('replay_dir: ~/custom_replay\n')
    config = get_config(config_with_tilde)
    assert '~' not in config['replay_dir']
    
    # Test 8: Custom abbreviations merged with defaults
    custom_abbrev = tmp_path / 'custom_abbrev.yaml'
    custom_abbrev.write_text('abbreviations:\n  custom: https://custom.com/{0}\n')
    config = get_config(custom_abbrev)
    assert 'gh' in config['abbreviations']
    assert 'custom' in config['abbreviations']
    assert config['abbreviations']['custom'] == 'https://custom.com/{0}'
    
    # Test 9: Default context is preserved
    config_with_context = tmp_path / 'context.yaml'
    config_with_context.write_text('default_context:\n  author: Test Author\n')
    config = get_config(config_with_context)
    assert config['default_context']['author'] == 'Test Author'


# LLM-generated content at query #17
#--------------------------

```python
import pytest
import os
import tempfile
from pathlib import Path
import yaml


def test_get_config():
    """Test get_config function with various scenarios."""
    
    # Test 1: Config file does not exist
    with pytest.raises(ConfigDoesNotExistException):
        get_config('/nonexistent/path/to/config.yaml')
    
    # Test 2: Valid config file with default values
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, 'test_config.yaml')
        config_data = {
            'cookiecutters_dir': '~/.cookiecutters/',
            'replay_dir': '~/.cookiecutter_replay/',
        }
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        result = get_config(config_path)
        assert isinstance(result, dict)
        assert 'cookiecutters_dir' in result
        assert 'replay_dir' in result
        assert result['cookiecutters_dir'] == os.path.expanduser('~/.cookiecutters/')
        assert result['replay_dir'] == os.path.expanduser('~/.cookiecutter_replay/')
    
    # Test 3: Config file with environment variables in paths
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, 'test_config.yaml')
        os.environ['TEST_DIR'] = tmpdir
        config_data = {
            'cookiecutters_dir': '$TEST_DIR/cookies',
            'replay_dir': '$TEST_DIR/replay',
        }
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        result = get_config(config_path)
        assert tmpdir in result['cookiecutters_dir']
        assert tmpdir in result['replay_dir']
    
    # Test 4: Config file with custom abbreviations
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, 'test_config.yaml')
        config_data = {
            'abbreviations': {
                'custom': 'https://custom.com/{0}.git'
            }
        }
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        result = get_config(config_path)
        assert 'abbreviations' in result
        assert 'custom' in result['abbreviations']
        assert 'gh' in result['abbreviations']  # Default abbreviations preserved
    
    # Test 5: Invalid YAML file
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, 'invalid_config.yaml')
        with open(config_path, 'w') as f:
            f.write('{ invalid yaml: [')
        
        with pytest.raises(InvalidConfiguration):
            get_config(config_path)
    
    # Test 6: YAML file with non-dict top-level element
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, 'list_config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(['item1', 'item2'], f)
        
        with pytest.raises(InvalidConfiguration):
            get_config(config_path)
    
    # Test 7: Empty YAML file
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, 'empty_config.yaml')
        with open(config_path, 'w') as f:
            f.write('')
        
        result = get_config(config_path)
        assert result == DEFAULT_CONFIG
    
    # Test 8: Config file with Path object
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / 'test_config.yaml'
        config_data = {'default_context': {'project_name': 'test'}}
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        result = get_config(config_path)
        assert isinstance(result, dict)
        assert 'default_context' in result
    
    # Test 9: Config file merges with defaults
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, 'partial_config.yaml')
        config_data = {'default_context': {'author': 'Test Author'}}
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        result = get_config(config_path)
        assert result['default_context']['author'] == 'Test Author'
        assert 'cookiecutters_dir' in result
        assert 'replay_dir' in result


# LLM-generated content at query #18
#--------------------------

```python
def test_get_user_config(tmp_path, monkeypatch):
    """Test get_user_config function with various scenarios."""
    
    # Test 1: Return default config when default_config is True
    config = get_user_config(default_config=True)
    assert config == DEFAULT_CONFIG
    assert config['cookiecutters_dir'] == os.path.expanduser('~/.cookiecutters/')
    assert config['replay_dir'] == os.path.expanduser('~/.cookiecutter_replay/')
    
    # Test 2: Merge provided dict with defaults when default_config is a dict
    custom_default = {'cookiecutters_dir': '/custom/path'}
    config = get_user_config(default_config=custom_default)
    assert config['cookiecutters_dir'] == '/custom/path'
    assert config['replay_dir'] == os.path.expanduser('~/.cookiecutter_replay/')
    
    # Test 3: Load custom config file when config_file is provided
    config_file = tmp_path / 'custom_config.yaml'
    config_file.write_text('cookiecutters_dir: /tmp/custom\nreplay_dir: /tmp/replay')
    config = get_user_config(config_file=str(config_file))
    assert config['cookiecutters_dir'] == '/tmp/custom'
    assert config['replay_dir'] == '/tmp/replay'
    
    # Test 4: Load from environment variable COOKIECUTTER_CONFIG
    env_config_file = tmp_path / 'env_config.yaml'
    env_config_file.write_text('cookiecutters_dir: /env/path')
    monkeypatch.setenv('COOKIECUTTER_CONFIG', str(env_config_file))
    monkeypatch.delenv('COOKIECUTTER_CONFIG', raising=False)
    monkeypatch.setenv('COOKIECUTTER_CONFIG', str(env_config_file))
    config = get_user_config()
    assert config['cookiecutters_dir'] == '/env/path'
    
    # Test 5: Return default config when no config file exists and no env var set
    monkeypatch.delenv('COOKIECUTTER_CONFIG', raising=False)
    monkeypatch.setattr('cookiecutter.config.USER_CONFIG_PATH', '/nonexistent/path')
    config = get_user_config()
    assert config == DEFAULT_CONFIG
    
    # Test 6: Load USER_CONFIG_PATH when it exists and no env var set
    monkeypatch.delenv('COOKIECUTTER_CONFIG', raising=False)
    user_config_file = tmp_path / '.cookiecutterrc'
    user_config_file.write_text('default_context:\n  author: Test Author')
    monkeypatch.setattr('cookiecutter.config.USER_CONFIG_PATH', str(user_config_file))
    config = get_user_config()
    assert config['default_context']['author'] == 'Test Author'
    
    # Test 7: Raise error when environment config file doesn't exist
    monkeypatch.delenv('COOKIECUTTER_CONFIG', raising=False)
    monkeypatch.setenv('COOKIECUTTER_CONFIG', '/nonexistent/env/config.yaml')
    monkeypatch.setattr('cookiecutter.config.USER_CONFIG_PATH', '/nonexistent/path')
    with pytest.raises(ConfigDoesNotExistException):
        get_user_config()
    
    # Test 8: Raise error when custom config file doesn't exist
    with pytest.raises(ConfigDoesNotExistException):
        get_user_config(config_file='/nonexistent/custom/config.yaml')
    
    # Test 9: default_config dict takes precedence over config_file
    config_file = tmp_path / 'config.yaml'
    config_file.write_text('cookiecutters_dir: /file/path')
    custom_dict = {'cookiecutters_dir': '/dict/path'}
    config = get_user_config(config_file=str(config_file), default_config=custom_dict)
    assert config['cookiecutters_dir'] == '/dict/path'
    
    # Test 10: Abbreviations are preserved from defaults
    config = get_user_config(default_config=True)
    assert 'gh' in config['abbreviations']
    assert config['abbreviations']['gh'] == 'https://github.com/{0}.git'


# LLM-generated content at query #19
#--------------------------

```python
def test_get_user_config(monkeypatch, tmp_path):
    """Test get_user_config function with various scenarios."""
    
    # Test 1: Return default config when default_config=True
    result = get_user_config(default_config=True)
    assert result == DEFAULT_CONFIG
    assert isinstance(result, dict)
    
    # Test 2: Merge provided dict with defaults when default_config is a dict
    custom_dict = {'cookiecutters_dir': '/custom/path'}
    result = get_user_config(default_config=custom_dict)
    assert result['cookiecutters_dir'] == '/custom/path'
    assert result['replay_dir'] == DEFAULT_CONFIG['replay_dir']
    
    # Test 3: Load custom config file
    config_file = tmp_path / 'custom_config.yml'
    config_file.write_text('cookiecutters_dir: /my/custom/path\n')
    result = get_user_config(config_file=str(config_file))
    assert result['cookiecutters_dir'] == '/my/custom/path'
    
    # Test 4: Load config from environment variable
    env_config_file = tmp_path / 'env_config.yml'
    env_config_file.write_text('replay_dir: /env/replay/path\n')
    monkeypatch.setenv('COOKIECUTTER_CONFIG', str(env_config_file))
    monkeypatch.delenv('COOKIECUTTER_CONFIG', raising=False)
    monkeypatch.setenv('COOKIECUTTER_CONFIG', str(env_config_file))
    result = get_user_config()
    assert '/env/replay/path' in result['replay_dir']
    
    # Test 5: Return defaults when no config file exists and no env var set
    monkeypatch.delenv('COOKIECUTTER_CONFIG', raising=False)
    monkeypatch.setattr('os.path.exists', lambda x: False)
    result = get_user_config()
    assert result == DEFAULT_CONFIG
    
    # Test 6: Load user config from default path if it exists
    monkeypatch.delenv('COOKIECUTTER_CONFIG', raising=False)
    default_config_file = tmp_path / '.cookiecutterrc'
    default_config_file.write_text('abbreviations:\n  custom: https://example.com/{0}\n')
    monkeypatch.setattr('cookiecutter.config.USER_CONFIG_PATH', str(default_config_file))
    monkeypatch.setattr('os.path.exists', lambda x: x == str(default_config_file))
    result = get_user_config()
    assert 'custom' in result['abbreviations']
    
    # Test 7: Raise error when config file doesn't exist
    monkeypatch.delenv('COOKIECUTTER_CONFIG', raising=False)
    with pytest.raises(ConfigDoesNotExistException):
        get_user_config(config_file='/nonexistent/path/config.yml')
    
    # Test 8: Raise error when env config file doesn't exist
    monkeypatch.setenv('COOKIECUTTER_CONFIG', '/nonexistent/env/config.yml')
    with pytest.raises(ConfigDoesNotExistException):
        get_user_config()
    
    # Test 9: Priority - default_config dict takes precedence
    monkeypatch.delenv('COOKIECUTTER_CONFIG', raising=False)
    custom_default = {'cookiecutters_dir': '/priority/path'}
    result = get_user_config(config_file='/some/file.yml', default_config=custom_default)
    assert result['cookiecutters_dir'] == '/priority/path'
    
    # Test 10: Priority - default_config=True takes precedence
    result = get_user_config(config_file='/some/file.yml', default_config=True)
    assert result == DEFAULT_CONFIG


# LLM-generated content at query #20
#--------------------------

```python
import pytest
import os
import tempfile
from pathlib import Path


def test_get_config():
    """Test get_config function with valid YAML configuration file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
        f.write("""
cookiecutters_dir: ~/.cookiecutters/
replay_dir: ~/.cookiecutter_replay/
default_context:
  project_name: My Project
abbreviations:
  gh: https://github.com/{0}.git
""")
        config_file = f.name

    try:
        config = get_config(config_file)
        
        assert isinstance(config, dict)
        assert 'cookiecutters_dir' in config
        assert 'replay_dir' in config
        assert 'default_context' in config
        assert 'abbreviations' in config
        assert config['default_context']['project_name'] == 'My Project'
    finally:
        os.unlink(config_file)


def test_get_config_file_not_exists():
    """Test get_config raises ConfigDoesNotExistException when file doesn't exist."""
    with pytest.raises(ConfigDoesNotExistException):
        get_config('/nonexistent/path/to/config.yml')


def test_get_config_invalid_yaml():
    """Test get_config raises InvalidConfiguration for invalid YAML."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
        f.write("invalid: yaml: content: [")
        config_file = f.name

    try:
        with pytest.raises(InvalidConfiguration):
            get_config(config_file)
    finally:
        os.unlink(config_file)


def test_get_config_non_dict_yaml():
    """Test get_config raises InvalidConfiguration when YAML top-level is not a dict."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
        f.write("- item1\n- item2\n")
        config_file = f.name

    try:
        with pytest.raises(InvalidConfiguration):
            get_config(config_file)
    finally:
        os.unlink(config_file)


def test_get_config_empty_file():
    """Test get_config with empty YAML file returns default config."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
        f.write("")
        config_file = f.name

    try:
        config = get_config(config_file)
        
        assert isinstance(config, dict)
        assert config['cookiecutters_dir'] == os.path.expanduser('~/.cookiecutters/')
        assert config['replay_dir'] == os.path.expanduser('~/.cookiecutter_replay/')
        assert config['abbreviations'] == BUILTIN_ABBREVIATIONS
    finally:
        os.unlink(config_file)


def test_get_config_path_expansion():
    """Test get_config expands environment variables and user home."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
        f.write("""
cookiecutters_dir: $HOME/.my_cookiecutters/
replay_dir: ~/my_replay/
""")
        config_file = f.name

    try:
        config = get_config(config_file)
        
        assert config['cookiecutters_dir'] == os.path.expanduser('~/.my_cookiecutters/')
        assert config['replay_dir'] == os.path.expanduser('~/my_replay/')
    finally:
        os.unlink(config_file)


def test_get_config_merges_with_defaults():
    """Test get_config merges provided config with default values."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
        f.write("""
cookiecutters_dir: ~/.custom_cookiecutters/
""")
        config_file = f.name

    try:
        config = get_config(config_file)
        
        assert config['cookiecutters_dir'] == os.path.expanduser('~/.custom_cookiecutters/')
        assert 'replay_dir' in config
        assert 'abbreviations' in config
        assert config['abbreviations']['gh'] == 'https://github.com/{0}.git'
    finally:
        os.unlink(config_file)


def test_get_config_custom_abbreviations():
    """Test get_config merges custom abbreviations with defaults."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
        f.write("""
abbreviations:
  custom: https://custom.example.com/{0}
""")
        config_file = f.name

    try:
        config = get_config(config_file)
        
        assert config['abbreviations']['custom'] == 'https://custom.example.com/{0}'
        assert config['abbreviations']['gh'] == 'https://github.com/{0}.git'
        assert config['abbreviations']['gl'] == 'https://gitlab.com/{0}.git'
    finally:
        os.unlink(config_file)


# LLM-generated content at query #21
#--------------------------

```python
def test_get_user_config(tmp_path, monkeypatch):
    """Test get_user_config function with various configurations."""
    
    # Test 1: Return defaults when default_config is True
    result = get_user_config(default_config=True)
    assert result == DEFAULT_CONFIG
    assert isinstance(result, dict)
    
    # Test 2: Merge custom dict with defaults when default_config is a dict
    custom_config = {'cookiecutters_dir': '/custom/path'}
    result = get_user_config(default_config=custom_config)
    assert result['cookiecutters_dir'] == '/custom/path'
    assert result['replay_dir'] == DEFAULT_CONFIG['replay_dir']
    assert result['abbreviations'] == DEFAULT_CONFIG['abbreviations']
    
    # Test 3: Load custom config file
    config_file = tmp_path / 'custom_config.yaml'
    config_content = 'cookiecutters_dir: /tmp/cookiecutters\nreplay_dir: /tmp/replay'
    config_file.write_text(config_content)
    result = get_user_config(config_file=str(config_file))
    assert result['cookiecutters_dir'] == '/tmp/cookiecutters'
    assert result['replay_dir'] == '/tmp/replay'
    
    # Test 4: Load from COOKIECUTTER_CONFIG environment variable
    monkeypatch.delenv('COOKIECUTTER_CONFIG', raising=False)
    monkeypatch.delenv('HOME', raising=False)
    env_config_file = tmp_path / 'env_config.yaml'
    env_config_file.write_text('cookiecutters_dir: /env/cookiecutters')
    monkeypatch.setenv('COOKIECUTTER_CONFIG', str(env_config_file))
    monkeypatch.setenv('HOME', str(tmp_path))
    result = get_user_config()
    assert result['cookiecutters_dir'] == '/env/cookiecutters'
    
    # Test 5: Return defaults when no config file exists and no env var set
    monkeypatch.delenv('COOKIECUTTER_CONFIG', raising=False)
    fake_home = tmp_path / 'fake_home'
    fake_home.mkdir()
    monkeypatch.setenv('HOME', str(fake_home))
    result = get_user_config()
    assert result == DEFAULT_CONFIG
    
    # Test 6: Load user config from default location if it exists
    monkeypatch.delenv('COOKIECUTTER_CONFIG', raising=False)
    user_config_path = fake_home / '.cookiecutterrc'
    user_config_path.write_text('replay_dir: /user/replay')
    result = get_user_config()
    assert result['replay_dir'] == '/user/replay'
    
    # Test 7: Merge nested dicts properly
    custom_dict = {
        'abbreviations': {'custom': 'https://custom.com/{0}'}
    }
    result = get_user_config(default_config=custom_dict)
    assert 'gh' in result['abbreviations']
    assert 'custom' in result['abbreviations']
    assert result['abbreviations']['custom'] == 'https://custom.com/{0}'
    
    # Test 8: Path expansion in loaded config
    config_file = tmp_path / 'path_expand_config.yaml'
    config_file.write_text('cookiecutters_dir: ~/expanded_path')
    result = get_user_config(config_file=str(config_file))
    assert '~' not in result['cookiecutters_dir']
    assert 'expanded_path' in result['cookiecutters_dir']
    
    # Test 9: Invalid config file raises exception
    invalid_config_file = tmp_path / 'invalid_config.yaml'
    invalid_config_file.write_text('invalid: yaml: content: [')
    with pytest.raises(InvalidConfiguration):
        get_user_config(config_file=str(invalid_config_file))
    
    # Test 10: Non-existent custom config file raises exception
    with pytest.raises(ConfigDoesNotExistException):
        get_user_config(config_file='/non/existent/path.yaml')


# LLM-generated content at query #22
#--------------------------

```python
def test_get_user_config(tmp_path, monkeypatch):
    """Test get_user_config function with various scenarios."""
    
    # Test 1: Return default config when default_config=True
    config = get_user_config(default_config=True)
    assert config == DEFAULT_CONFIG
    assert 'cookiecutters_dir' in config
    assert 'replay_dir' in config
    assert 'default_context' in config
    assert 'abbreviations' in config
    
    # Test 2: Merge provided dict with defaults when default_config is a dict
    custom_dict = {'cookiecutters_dir': '/custom/path'}
    config = get_user_config(default_config=custom_dict)
    assert config['cookiecutters_dir'] == '/custom/path'
    assert config['replay_dir'] == DEFAULT_CONFIG['replay_dir']
    assert config['abbreviations'] == DEFAULT_CONFIG['abbreviations']
    
    # Test 3: Load custom config file
    config_file = tmp_path / 'custom_config.yaml'
    config_file.write_text('cookiecutters_dir: /tmp/custom\nreplay_dir: /tmp/replay\n')
    config = get_user_config(config_file=str(config_file))
    assert '/tmp/custom' in config['cookiecutters_dir']
    assert '/tmp/replay' in config['replay_dir']
    
    # Test 4: Load from COOKIECUTTER_CONFIG environment variable
    env_config_file = tmp_path / 'env_config.yaml'
    env_config_file.write_text('cookiecutters_dir: /env/path\n')
    monkeypatch.setenv('COOKIECUTTER_CONFIG', str(env_config_file))
    monkeypatch.delenv('COOKIECUTTER_CONFIG', raising=False)
    monkeypatch.setenv('COOKIECUTTER_CONFIG', str(env_config_file))
    config = get_user_config()
    assert '/env/path' in config['cookiecutters_dir']
    
    # Test 5: Return defaults when no config file exists and no env var set
    monkeypatch.delenv('COOKIECUTTER_CONFIG', raising=False)
    monkeypatch.setattr('os.path.exists', lambda x: False)
    config = get_user_config()
    assert config == DEFAULT_CONFIG
    
    # Test 6: Load USER_CONFIG_PATH if it exists
    monkeypatch.delenv('COOKIECUTTER_CONFIG', raising=False)
    user_config_file = tmp_path / 'user_config.yaml'
    user_config_file.write_text('cookiecutters_dir: /user/path\n')
    monkeypatch.setattr('cookiecutter.config.USER_CONFIG_PATH', str(user_config_file))
    config = get_user_config()
    assert '/user/path' in config['cookiecutters_dir']
    
    # Test 7: Raise exception when config file doesn't exist
    non_existent = tmp_path / 'non_existent.yaml'
    with pytest.raises(ConfigDoesNotExistException):
        get_user_config(config_file=str(non_existent))
    
    # Test 8: Raise exception for invalid YAML
    invalid_yaml = tmp_path / 'invalid.yaml'
    invalid_yaml.write_text('invalid: yaml: content:')
    with pytest.raises(InvalidConfiguration):
        get_user_config(config_file=str(invalid_yaml))
    
    # Test 9: Raise exception when YAML top-level is not a dict
    non_dict_yaml = tmp_path / 'non_dict.yaml'
    non_dict_yaml.write_text('- item1\n- item2\n')
    with pytest.raises(InvalidConfiguration):
        get_user_config(config_file=str(non_dict_yaml))
    
    # Test 10: Priority - default_config=True takes precedence over config_file
    config = get_user_config(config_file=str(config_file), default_config=True)
    assert config == DEFAULT_CONFIG


# LLM-generated content at query #23
#--------------------------

```python
import pytest
import os
import tempfile
import yaml
from pathlib import Path


def test_get_config():
    """Test get_config function with valid YAML configuration file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_file = os.path.join(tmpdir, 'cookiecutterrc')
        config_data = {
            'cookiecutters_dir': '~/.cookiecutters/',
            'replay_dir': '~/.cookiecutter_replay/',
            'default_context': {},
            'abbreviations': {
                'gh': 'https://github.com/{0}.git',
            }
        }
        
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f)
        
        result = get_config(config_file)
        
        assert isinstance(result, dict)
        assert 'cookiecutters_dir' in result
        assert 'replay_dir' in result
        assert os.path.expanduser('~/.cookiecutters/') == result['cookiecutters_dir']
        assert os.path.expanduser('~/.cookiecutter_replay/') == result['replay_dir']


def test_get_config_file_not_exists():
    """Test get_config raises exception when config file does not exist."""
    with pytest.raises(ConfigDoesNotExistException):
        get_config('/nonexistent/path/to/config')


def test_get_config_invalid_yaml():
    """Test get_config raises exception for invalid YAML."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_file = os.path.join(tmpdir, 'cookiecutterrc')
        
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write('invalid: yaml: content: [')
        
        with pytest.raises(InvalidConfiguration):
            get_config(config_file)


def test_get_config_non_dict_top_level():
    """Test get_config raises exception when top-level YAML is not a dict."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_file = os.path.join(tmpdir, 'cookiecutterrc')
        
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(['list', 'not', 'dict'], f)
        
        with pytest.raises(InvalidConfiguration):
            get_config(config_file)


def test_get_config_with_env_vars():
    """Test get_config expands environment variables in paths."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_file = os.path.join(tmpdir, 'cookiecutterrc')
        config_data = {
            'cookiecutters_dir': '$HOME/.cookiecutters/',
            'replay_dir': '${HOME}/.cookiecutter_replay/',
        }
        
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f)
        
        result = get_config(config_file)
        
        assert '$HOME' not in result['cookiecutters_dir']
        assert '${HOME}' not in result['replay_dir']


def test_get_config_merges_with_defaults():
    """Test get_config merges loaded config with default config."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_file = os.path.join(tmpdir, 'cookiecutterrc')
        config_data = {
            'cookiecutters_dir': '/custom/path/',
        }
        
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f)
        
        result = get_config(config_file)
        
        assert result['cookiecutters_dir'] == '/custom/path/'
        assert 'replay_dir' in result
        assert 'abbreviations' in result
        assert 'gh' in result['abbreviations']


def test_get_config_empty_file():
    """Test get_config with empty YAML file returns defaults."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_file = os.path.join(tmpdir, 'cookiecutterrc')
        
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write('')
        
        result = get_config(config_file)
        
        assert isinstance(result, dict)
        assert result['cookiecutters_dir'] == os.path.expanduser('~/.cookiecutters/')
        assert result['replay_dir'] == os.path.expanduser('~/.cookiecutter_replay/')


def test_get_config_nested_abbreviations():
    """Test get_config preserves and merges nested abbreviations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_file = os.path.join(tmpdir, 'cookiecutterrc')
        config_data = {
            'abbreviations': {
                'custom': 'https://custom.com/{0}.git',
            }
        }
        
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f)
        
        result = get_config(config_file)
        
        assert 'gh' in result['abbreviations']
        assert 'custom' in result['abbreviations']


# LLM-generated content at query #24
#--------------------------

```python
import pytest
import os
import tempfile
from pathlib import Path
import yaml


def test_get_config():
    """Test get_config function with various scenarios."""
    
    # Test 1: Config file does not exist
    with pytest.raises(ConfigDoesNotExistException):
        get_config('/nonexistent/path/to/config.yaml')
    
    # Test 2: Valid config file with all fields
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        config_data = {
            'cookiecutters_dir': '~/.custom_cookiecutters/',
            'replay_dir': '~/.custom_replay/',
            'default_context': {'key': 'value'},
            'abbreviations': {'custom': 'https://example.com/{0}'}
        }
        yaml.dump(config_data, f)
        f.flush()
        temp_path = f.name
    
    try:
        config = get_config(temp_path)
        assert config['cookiecutters_dir'] == os.path.expanduser('~/.custom_cookiecutters/')
        assert config['replay_dir'] == os.path.expanduser('~/.custom_replay/')
        assert config['default_context'] == {'key': 'value'}
        assert 'custom' in config['abbreviations']
        assert 'gh' in config['abbreviations']  # Built-in abbreviations preserved
    finally:
        os.unlink(temp_path)
    
    # Test 3: Config file with environment variables in paths
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        config_data = {
            'cookiecutters_dir': '$HOME/.cookiecutters/',
            'replay_dir': '$HOME/.replay/'
        }
        yaml.dump(config_data, f)
        f.flush()
        temp_path = f.name
    
    try:
        config = get_config(temp_path)
        assert '$HOME' not in config['cookiecutters_dir']
        assert '$HOME' not in config['replay_dir']
    finally:
        os.unlink(temp_path)
    
    # Test 4: Empty config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write('')
        f.flush()
        temp_path = f.name
    
    try:
        config = get_config(temp_path)
        assert 'cookiecutters_dir' in config
        assert 'replay_dir' in config
        assert 'abbreviations' in config
    finally:
        os.unlink(temp_path)
    
    # Test 5: Invalid YAML syntax
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write('invalid: yaml: content: [')
        f.flush()
        temp_path = f.name
    
    try:
        with pytest.raises(InvalidConfiguration):
            get_config(temp_path)
    finally:
        os.unlink(temp_path)
    
    # Test 6: YAML file with non-dict top-level element
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(['list', 'not', 'dict'], f)
        f.flush()
        temp_path = f.name
    
    try:
        with pytest.raises(InvalidConfiguration):
            get_config(temp_path)
    finally:
        os.unlink(temp_path)
    
    # Test 7: Config with Path object as input
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        config_data = {'cookiecutters_dir': '~/.custom/'}
        yaml.dump(config_data, f)
        f.flush()
        temp_path = Path(f.name)
    
    try:
        config = get_config(temp_path)
        assert 'cookiecutters_dir' in config
    finally:
        os.unlink(temp_path)
    
    # Test 8: Partial config merges with defaults
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        config_data = {'cookiecutters_dir': '~/.my_cookiecutters/'}
        yaml.dump(config_data, f)
        f.flush()
        temp_path = f.name
    
    try:
        config = get_config(temp_path)
        assert config['cookiecutters_dir'] == os.path.expanduser('~/.my_cookiecutters/')
        assert 'replay_dir' in config
        assert config['abbreviations'] == BUILTIN_ABBREVIATIONS
    finally:
        os.unlink(temp_path)


# LLM-generated content at query #25
#--------------------------

```python
def test_get_config(tmp_path):
    """Test get_config function with valid and invalid configurations."""
    # Test 1: Valid YAML config file
    config_file = tmp_path / "valid_config.yaml"
    config_content = """
cookiecutters_dir: /tmp/cookiecutters
replay_dir: /tmp/replay
default_context:
  full_name: John Doe
abbreviations:
  gh: https://github.com/{0}.git
"""
    config_file.write_text(config_content)
    
    result = get_config(config_file)
    
    assert result['cookiecutters_dir'] == '/tmp/cookiecutters'
    assert result['replay_dir'] == '/tmp/replay'
    assert result['default_context']['full_name'] == 'John Doe'
    assert result['abbreviations']['gh'] == 'https://github.com/{0}.git'
    # Should also have builtin abbreviations merged
    assert result['abbreviations']['gl'] == 'https://gitlab.com/{0}.git'

    # Test 2: Config file with environment variable expansion
    config_file2 = tmp_path / "env_config.yaml"
    config_content2 = """
cookiecutters_dir: $HOME/.cookiecutters
replay_dir: ~/replay
"""
    config_file2.write_text(config_content2)
    
    result2 = get_config(config_file2)
    
    assert '$HOME' not in result2['cookiecutters_dir']
    assert '~' not in result2['replay_dir']

    # Test 3: Empty YAML file (should use defaults)
    config_file3 = tmp_path / "empty_config.yaml"
    config_file3.write_text("")
    
    result3 = get_config(config_file3)
    
    assert 'cookiecutters_dir' in result3
    assert 'replay_dir' in result3
    assert 'default_context' in result3

    # Test 4: Non-existent config file
    non_existent = tmp_path / "nonexistent.yaml"
    
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent)

    # Test 5: Invalid YAML syntax
    invalid_yaml = tmp_path / "invalid.yaml"
    invalid_yaml.write_text("{ invalid yaml: [")
    
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml)

    # Test 6: YAML with non-dict top-level element
    non_dict_yaml = tmp_path / "non_dict.yaml"
    non_dict_yaml.write_text("- item1\n- item2")
    
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml)

    # Test 7: Merging with defaults preserves abbreviations
    config_file4 = tmp_path / "partial_config.yaml"
    config_content4 = """
default_context:
  author_name: Jane Smith
"""
    config_file4.write_text(config_content4)
    
    result4 = get_config(config_file4)
    
    assert result4['default_context']['author_name'] == 'Jane Smith'
    assert 'gh' in result4['abbreviations']
    assert 'gl' in result4['abbreviations']
    assert 'bb' in result4['abbreviations']


# LLM-generated content at query #26
#--------------------------

```python
def test_get_config(tmp_path, monkeypatch):
    """Test get_config function with various scenarios."""
    # Test 1: Non-existent config file raises ConfigDoesNotExistException
    with pytest.raises(ConfigDoesNotExistException):
        get_config('/nonexistent/path/config.yaml')

    # Test 2: Valid YAML config file is loaded correctly
    config_file = tmp_path / 'config.yaml'
    config_content = {
        'cookiecutters_dir': '~/.cookiecutters/',
        'replay_dir': '~/.cookiecutter_replay/',
        'default_context': {'project_name': 'my_project'},
        'abbreviations': {'custom': 'https://example.com/{0}.git'},
    }
    config_file.write_text(yaml.dump(config_content), encoding='utf-8')
    result = get_config(config_file)
    
    assert isinstance(result, dict)
    assert 'cookiecutters_dir' in result
    assert 'replay_dir' in result
    assert result['default_context']['project_name'] == 'my_project'
    assert result['abbreviations']['custom'] == 'https://example.com/{0}.git'
    assert result['abbreviations']['gh'] == 'https://github.com/{0}.git'

    # Test 3: Invalid YAML raises InvalidConfiguration
    invalid_yaml_file = tmp_path / 'invalid.yaml'
    invalid_yaml_file.write_text('invalid: yaml: content:', encoding='utf-8')
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml_file)

    # Test 4: Non-dict top-level element raises InvalidConfiguration
    non_dict_file = tmp_path / 'non_dict.yaml'
    non_dict_file.write_text('- item1\n- item2', encoding='utf-8')
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_file)

    # Test 5: Empty YAML file returns default config
    empty_file = tmp_path / 'empty.yaml'
    empty_file.write_text('', encoding='utf-8')
    result = get_config(empty_file)
    assert result['cookiecutters_dir'] == os.path.expanduser('~/.cookiecutters/')
    assert result['replay_dir'] == os.path.expanduser('~/.cookiecutter_replay/')

    # Test 6: Path expansion works correctly
    config_with_vars = tmp_path / 'config_vars.yaml'
    config_with_vars.write_text(
        'cookiecutters_dir: $HOME/.my_cookiecutters/\nreplay_dir: ~/replay/',
        encoding='utf-8'
    )
    result = get_config(config_with_vars)
    assert '$HOME' not in result['cookiecutters_dir']
    assert '~' not in result['replay_dir']

    # Test 7: Partial config merges with defaults
    partial_config_file = tmp_path / 'partial.yaml'
    partial_config_file.write_text('default_context:\n  key: value', encoding='utf-8')
    result = get_config(partial_config_file)
    assert 'cookiecutters_dir' in result
    assert 'replay_dir' in result
    assert result['default_context']['key'] == 'value'


# LLM-generated content at query #27
#--------------------------

```python
def test_get_config(tmp_path):
    """Test get_config function with valid and invalid configurations."""
    import pytest
    from pathlib import Path
    
    # Test 1: Valid YAML config file
    config_file = tmp_path / "config.yaml"
    config_content = """
cookiecutters_dir: ~/.cookiecutters/
replay_dir: ~/.cookiecutter_replay/
default_context:
  author_name: John Doe
abbreviations:
  custom: https://custom.com/{0}
"""
    config_file.write_text(config_content)
    result = get_config(config_file)
    
    assert isinstance(result, dict)
    assert 'cookiecutters_dir' in result
    assert 'replay_dir' in result
    assert 'default_context' in result
    assert 'abbreviations' in result
    assert result['default_context']['author_name'] == 'John Doe'
    assert 'custom' in result['abbreviations']
    assert 'gh' in result['abbreviations']  # Built-in abbreviations preserved
    
    # Test 2: Config file with environment variables
    config_file2 = tmp_path / "config2.yaml"
    config_content2 = """
cookiecutters_dir: $HOME/.cookiecutters/
replay_dir: ~/custom_replay/
"""
    config_file2.write_text(config_content2)
    result2 = get_config(config_file2)
    
    assert os.path.expandvars('$HOME') in result2['cookiecutters_dir']
    
    # Test 3: Non-existent config file
    non_existent = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent)
    
    # Test 4: Invalid YAML syntax
    invalid_yaml = tmp_path / "invalid.yaml"
    invalid_yaml.write_text("invalid: yaml: content: [")
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml)
    
    # Test 5: YAML with non-dict top level
    non_dict_yaml = tmp_path / "non_dict.yaml"
    non_dict_yaml.write_text("- item1\n- item2")
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml)
    
    # Test 6: Empty YAML file
    empty_yaml = tmp_path / "empty.yaml"
    empty_yaml.write_text("")
    result6 = get_config(empty_yaml)
    
    assert result6 == DEFAULT_CONFIG
    
    # Test 7: Partial config merges with defaults
    partial_yaml = tmp_path / "partial.yaml"
    partial_yaml.write_text("cookiecutters_dir: /custom/path/")
    result7 = get_config(partial_yaml)
    
    assert result7['cookiecutters_dir'] == '/custom/path/'
    assert 'replay_dir' in result7
    assert result7['abbreviations'] == BUILTIN_ABBREVIATIONS


# LLM-generated content at query #28
#--------------------------

```python
def test_get_config(tmp_path, monkeypatch):
    """Test get_config function with various scenarios."""
    
    # Test 1: Config file does not exist
    non_existent_path = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException):
        get_config(non_existent_path)
    
    # Test 2: Valid YAML config file
    config_file = tmp_path / "config.yaml"
    config_content = {
        'cookiecutters_dir': '/custom/cookiecutters',
        'replay_dir': '/custom/replay',
        'default_context': {'key': 'value'},
        'abbreviations': {'custom': 'https://example.com/{0}.git'}
    }
    config_file.write_text(yaml.dump(config_content))
    
    result = get_config(config_file)
    assert result['cookiecutters_dir'] == '/custom/cookiecutters'
    assert result['replay_dir'] == '/custom/replay'
    assert result['default_context']['key'] == 'value'
    assert result['abbreviations']['custom'] == 'https://example.com/{0}.git'
    # Check that builtin abbreviations are preserved
    assert result['abbreviations']['gh'] == 'https://github.com/{0}.git'
    
    # Test 3: Path expansion with environment variables and home
    config_with_vars = tmp_path / "config_vars.yaml"
    config_with_vars.write_text(yaml.dump({
        'cookiecutters_dir': '~/.cookiecutters',
        'replay_dir': '$HOME/.cookiecutter_replay'
    }))
    
    result = get_config(config_with_vars)
    assert result['cookiecutters_dir'] == os.path.expanduser('~/.cookiecutters')
    assert '$HOME' not in result['replay_dir']
    
    # Test 4: Invalid YAML syntax
    invalid_yaml = tmp_path / "invalid.yaml"
    invalid_yaml.write_text("{ invalid yaml content: [")
    
    with pytest.raises(InvalidConfiguration):
        get_config(invalid_yaml)
    
    # Test 5: YAML file with non-dict top-level element
    non_dict_yaml = tmp_path / "non_dict.yaml"
    non_dict_yaml.write_text("- item1\n- item2")
    
    with pytest.raises(InvalidConfiguration):
        get_config(non_dict_yaml)
    
    # Test 6: Empty YAML file
    empty_yaml = tmp_path / "empty.yaml"
    empty_yaml.write_text("")
    
    result = get_config(empty_yaml)
    assert result == DEFAULT_CONFIG
    
    # Test 7: Partial config (only some keys)
    partial_config = tmp_path / "partial.yaml"
    partial_config.write_text(yaml.dump({
        'cookiecutters_dir': '/partial/path'
    }))
    
    result = get_config(partial_config)
    assert result['cookiecutters_dir'] == '/partial/path'
    assert 'replay_dir' in result  # Should have default value
    assert result['replay_dir'] == os.path.expanduser('~/.cookiecutter_replay/')
    
    # Test 8: Config with string path argument
    string_config = tmp_path / "string_path.yaml"
    string_config.write_text(yaml.dump({'cookiecutters_dir': '/test'}))
    
    result = get_config(str(string_config))
    assert result['cookiecutters_dir'] == '/test'


# LLM-generated content at query #29
#--------------------------

```python
def test_get_config(tmp_path, monkeypatch):
    """Test get_config function with valid and invalid configurations."""
    # Test 1: Valid config file
    config_file = tmp_path / "config.yaml"
    config_content = """
cookiecutters_dir: /custom/cookiecutters
replay_dir: /custom/replay
default_context:
    project_name: my_project
abbreviations:
    custom: https://example.com/{0}.git
"""
    config_file.write_text(config_content)
    
    result = get_config(config_file)
    
    assert result['cookiecutters_dir'] == '/custom/cookiecutters'
    assert result['replay_dir'] == '/custom/replay'
    assert result['default_context']['project_name'] == 'my_project'
    assert result['abbreviations']['custom'] == 'https://example.com/{0}.git'
    # Builtin abbreviations should be preserved
    assert result['abbreviations']['gh'] == 'https://github.com/{0}.git'


def test_get_config_nonexistent_file(tmp_path):
    """Test get_config raises exception for non-existent file."""
    config_file = tmp_path / "nonexistent.yaml"
    
    with pytest.raises(ConfigDoesNotExistException):
        get_config(config_file)


def test_get_config_invalid_yaml(tmp_path):
    """Test get_config raises exception for invalid YAML."""
    config_file = tmp_path / "invalid.yaml"
    config_file.write_text("invalid: yaml: content: [")
    
    with pytest.raises(InvalidConfiguration):
        get_config(config_file)


def test_get_config_non_dict_yaml(tmp_path):
    """Test get_config raises exception when YAML is not a dict."""
    config_file = tmp_path / "list.yaml"
    config_file.write_text("- item1\n- item2")
    
    with pytest.raises(InvalidConfiguration):
        get_config(config_file)


def test_get_config_empty_file(tmp_path):
    """Test get_config with empty YAML file returns defaults."""
    config_file = tmp_path / "empty.yaml"
    config_file.write_text("")
    
    result = get_config(config_file)
    
    assert result['cookiecutters_dir'] == os.path.expanduser('~/.cookiecutters/')
    assert result['replay_dir'] == os.path.expanduser('~/.cookiecutter_replay/')


def test_get_config_expands_paths(tmp_path):
    """Test get_config expands environment variables and user home."""
    config_file = tmp_path / "config.yaml"
    config_content = """
cookiecutters_dir: ~/my_cookiecutters
replay_dir: $HOME/.replay
"""
    config_file.write_text(config_content)
    
    result = get_config(config_file)
    
    assert result['cookiecutters_dir'] == os.path.expanduser('~/my_cookiecutters')
    assert '~' not in result['cookiecutters_dir']
    assert '$HOME' not in result['replay_dir']


def test_get_config_partial_config(tmp_path):
    """Test get_config merges partial config with defaults."""
    config_file = tmp_path / "partial.yaml"
    config_content = """
default_context:
    author: John Doe
"""
    config_file.write_text(config_content)
    
    result = get_config(config_file)
    
    assert result['default_context']['author'] == 'John Doe'
    assert result['cookiecutters_dir'] == os.path.expanduser('~/.cookiecutters/')
    assert result['abbreviations']['gh'] == 'https://github.com/{0}.git'


# LLM-generated content at query #30
#--------------------------

```python
def test_get_user_config(tmp_path, monkeypatch):
    """Test get_user_config function with various scenarios."""
    
    # Test 1: default_config is True - should return default config
    result = get_user_config(default_config=True)
    assert result == DEFAULT_CONFIG
    assert isinstance(result, dict)
    
    # Test 2: default_config is a dict - should merge with defaults
    custom_default = {'cookiecutters_dir': '/custom/path'}
    result = get_user_config(default_config=custom_default)
    assert result['cookiecutters_dir'] == '/custom/path'
    assert result['replay_dir'] == DEFAULT_CONFIG['replay_dir']
    assert 'abbreviations' in result
    
    # Test 3: Load custom config file
    config_file = tmp_path / "custom_config.yml"
    config_file.write_text("cookiecutters_dir: /custom/cookiecutters\n")
    result = get_user_config(config_file=str(config_file))
    assert result['cookiecutters_dir'] == '/custom/cookiecutters'
    
    # Test 4: Environment variable set with valid config
    env_config_file = tmp_path / "env_config.yml"
    env_config_file.write_text("replay_dir: /env/replay\n")
    monkeypatch.setenv('COOKIECUTTER_CONFIG', str(env_config_file))
    monkeypatch.delenv('COOKIECUTTER_CONFIG', raising=False)
    monkeypatch.setenv('COOKIECUTTER_CONFIG', str(env_config_file))
    result = get_user_config()
    assert result['replay_dir'] == '/env/replay'
    
    # Test 5: Environment variable not set, user config doesn't exist
    monkeypatch.delenv('COOKIECUTTER_CONFIG', raising=False)
    monkeypatch.setenv('HOME', str(tmp_path))
    result = get_user_config()
    assert result == DEFAULT_CONFIG
    
    # Test 6: Environment variable not set, user config exists
    user_config_path = tmp_path / ".cookiecutterrc"
    user_config_path.write_text("abbreviations:\n  custom: 'https://custom.com/{0}'\n")
    monkeypatch.delenv('COOKIECUTTER_CONFIG', raising=False)
    with monkeypatch.context() as mp:
        mp.setattr('cookiecutter.config.USER_CONFIG_PATH', str(user_config_path))
        result = get_user_config()
        assert 'abbreviations' in result
    
    # Test 7: default_config dict takes precedence over config_file
    result = get_user_config(
        config_file=str(config_file),
        default_config={'custom_key': 'custom_value'}
    )
    assert result['custom_key'] == 'custom_value'
    assert 'cookiecutters_dir' in result
    
    # Test 8: Config file with invalid YAML should raise InvalidConfiguration
    invalid_config = tmp_path / "invalid.yml"
    invalid_config.write_text("invalid: yaml: content: :\n")
    with pytest.raises(InvalidConfiguration):
        get_user_config(config_file=str(invalid_config))
    
    # Test 9: Non-existent config file should raise ConfigDoesNotExistException
    with pytest.raises(ConfigDoesNotExistException):
        get_user_config(config_file="/non/existent/path.yml")
    
    # Test 10: Config file that is not USER_CONFIG_PATH should be loaded
    custom_path = tmp_path / "my_config.yml"
    custom_path.write_text("cookiecutters_dir: /my/path\n")
    result = get_user_config(config_file=str(custom_path))
    assert result['cookiecutters_dir'] == '/my/path'


