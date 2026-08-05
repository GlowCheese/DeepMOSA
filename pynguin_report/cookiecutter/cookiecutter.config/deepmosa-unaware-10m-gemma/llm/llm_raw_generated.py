####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import os
import pytest
from unittest.mock import patch, mock_open
from pathlib import Path

def test_get_user_config(tmp_path):
    # Test Case 1: default_config is True (return defaults)
    config = get_user_config(default_config=True)
    assert config == DEFAULT_CONFIG
    assert config != DEFAULT_CONFIG  # Ensure it's a copy, not the same object

    # Test Case 2: default_config is a dict (merge with defaults)
    custom_overrides = {'replay_dir': '/tmp/custom_replay/', 'new_key': 'new_val'}
    config = get_all_configs_with_override(custom_overrides) # Helper logic below
    # Manual implementation for clarity in test:
    config = merge_configs(DEFAULT_CONFIG, custom_overrides)
    assert config['replay_dir'] == '/tmp/custom_replay/'
    assert config['new_key'] == 'new_val'
    assert config['abbreviations'] == BUILTIN_ABBREVIATIONS

    # Test Case 3: config_file provided and valid
    yaml_content = "cookiecutters_dir: '/tmp/custom_cc/'\nabbreviations:\n  new: 'http://new.com/{0}'"
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml_content)
    
    config = get_user_config(config_file=str(config_file))
    assert config['cookiecutters_dir'] == os.path.expanduser('/tmp/custom_cc/')
    assert config['abbreviations']['new'] == 'http://new.com/{0}'
    # Ensure old abbreviations are preserved via merge_configs logic in get_config
    assert config['abbreviations']['gh'] == BUILTIN_ABBREVIATIONS['gh']

    # Test Case 4: config_file provided but does not exist
    with pytest.raises(ConfigDoesNotExistException):
        get_user_config(config_file="non_existent_path.yaml")

    # Test Case 5: COOKIECUTTER_CONFIG env var is set
    custom_env_config = tmp_path / "env_config.yaml"
    custom_env_config.write_text("replay_dir: '/tmp/env_replay/'")
    
    with patch.dict(os.environ, {'COOKIECUTTER_CONFIG': str(custom_env_config)}):
        config = get_user_config()
        assert config['replay_dir'] == os.path.expanduser('/tmp/env_replay/')

    # Test Case 6: COOKIECUTTER_CONFIG env var is NOT set, and USER_CONFIG_PATH does not exist
    with patch.dict(os.environ, {}, clear=True):
        with patch('os.path.exists', side_effect=lambda p: p != USER_CONFIG_PATH):
            config = get_user_config()
            assert config == DEFAULT_CONFIG

    # Test Case 7: COOKIECUTTER_CONFIG env var is NOT set, and USER_CONFIG_PATH exists
    with patch.dict(os.environ, {}, clear=True):
        user_cfg_content = "cookiecutters_dir: '/tmp/user_cc/'"
        # We mock the file content of the default user config path
        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=user_cfg_content)):
                config = get_user_config()
                assert config['cookiecutters_dir'] == os.path.expanduser('/tmp/user_cc/')

    # Test Case 8: Invalid YAML in config file
    invalid_yaml = "key: : value" # Broken YAML
    bad_config = tmp_path / "bad_config.yaml"
    bad_config.write_text(invalid_yaml)
    with pytest.raises(InvalidConfiguration):
        get_user_config(config_file=str(bad_config))

# Helper to allow testing the merge logic specifically via get_user_config's param path
def get_all_configs_with_override(overrides):
    return get_user_config(default_config=overrides)
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest
import yaml
from pathlib import Path
from unittest.mock import patch, mock_open
from cookiecutter.exceptions import ConfigDoesNotExistException, InvalidConfiguration

def test_get_config(tmp_path):
    # Test 1: File does not exist
    non_existent_path = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException) as excinfo:
        get_config(non_existent_path)
    assert "does not exist" in str(excinfo.value)

    # Test 2: Valid YAML file
    valid_yaml_content = {
        'cookiecutments_dir': '/tmp/custom_dir',
        'abbreviations': {'gh': 'https://github.com/custom'}
    }
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump(valid_yaml_content))

    config = get_config(config_file)
    
    # Check merged values (abbreviations should be merged, not overwritten entirely)
    assert config['abbreviations']['gh'] == 'https://github.com/custom'
    assert config['abbreviations']['gl'] == 'https://gitlab.com/{0}.git'
    # Check override
    assert config['cookiecutters_dir'].endswith('custom_dir')

    # Test 3: Invalid YAML syntax
    invalid_yaml_file = tmp_path / "invalid.yaml"
    invalid_yaml_file.write_text("key: : value") # Syntax error
    with pytest.raises(InvalidConfiguration) as excinfo:
        get_config(invalid_yaml_file)
    assert "Unable to parse YAML file" in str(excinfo.value)

    # Test 4: YAML is not a dictionary (e.g., a list)
    list_yaml_file = tmp_path / "list.yaml"
    list_yaml_file.write_text("- item1\n- item2")
    with pytest.raises(InvalidConfiguration) as excinfo:
        get_config(list_yaml_file)
    assert "should be an object" in str(excinfo.value)

    # Test 5: Empty YAML file (should return DEFAULT_CONFIG with expanded paths)
    empty_yaml_file = tmp_path / "empty.yaml"
    empty_yaml_file.write_text("")
    config = get_config(empty_yaml_file)
    assert config['replay_dir'] == _expand_path(DEFAULT_CONFIG['replay_dir'])
    assert config['abbreviations']['bb'] == 'https://bitbucket.org/{0}'

def test_get_config_expansion(tmp_path):
    # Test path expansion for environment variables
    config_file = tmp_path / "env_test.yaml"
    content = {
        'cookiecutters_dir': '$HOME/my_cookies',
        'replay_dir': '~/my_replay'
    }
    config_file.write_text(yaml.dump(content))
    
    with patch.dict('os.environ', {'HOME': '/home/testuser'}):
        config = get_config(config_file)
        assert config['cookiecutters_dir'] == '/home/testuser/my_cookies'
        assert config['replay_dir'] == '/home/testuser/my_replay'
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest
import yaml
from pathlib import Path
from unittest.mock import patch, mock_open
from cookiecutter.exceptions import ConfigDoesNotExistException, InvalidConfiguration

def test_get_config(tmp_path):
    # Test Case 1: File does not exist
    non_existent_path = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException) as excinfo:
        get_config(non_existent_path)
    assert "does not exist" in str(excinfo.value)

    # Test Case 2: Invalid YAML syntax
    invalid_yaml_path = tmp_path / "invalid.yaml"
    invalid_yaml_path.write_text("key: : value", encoding="utf-8")
    with pytest.raises(InvalidConfiguration) as excinfo:
        get_config(invalid_yaml_path)
    assert "Unable to parse YAML file" in str(excinfo.value)

    # Test Case 3: Top-level element is not a dictionary (e.g., a list)
    list_yaml_path = tmp_path / "list.yaml"
    list_yaml_path.write_text("- item1\n- item2", encoding="utf-8")
    with pytest.raises(InvalidConfiguration) as excinfo:
        get_config(list_yaml_path)
    assert "should be an object" in str(excinfo.value)

    # Test Case 4: Valid configuration loading and merging
    valid_yaml_path = tmp_path / "valid.yaml"
    custom_content = {
        'cookiecutters_dir': '/tmp/custom_cookies/',
        'abbreviations': {
            'new_abbr': 'https://example.com/{0}'
        }
    }
    valid_yaml_path.write_text(yaml.dump(custom_content), encoding="utf-8")

    config = get_config(valid_yaml_path)

    # Verify merging logic: existing keys in nested dicts (abbreviations) should be preserved
    assert config['abbreviations']['gh'] == 'https://github.com/{0}.git'
    assert config['abbreviations']['new_abbr'] == 'https://example.com/{0}'
    
    # Verify overriding logic: top level keys should be updated
    assert config['cookiecutters_dir'] == '/tmp/custom_cookies/'
    
    # Verify default values remain for untouched keys
    assert 'replay_dir' in config
    assert isinstance(config['default_context'], dict)

    # Test Case 5: Path expansion logic via _expand_path integration
    env_var_yaml_path = tmp_path / "env_var.yaml"
    env_var_content = {
        'replay_dir': '$HOME/replay_test/'
    }
    env_var_yaml_path.write_text(yaml.dump(env_var_content), encoding="utf-8")
    
    # Mock os.environ to ensure $HOME resolves predictably for the test
    with patch.dict('os.environ', {'HOME': '/tmp/user'}):
        config_env = get_config(env_var_yaml_path)
        # Should expand $HOME to /tmp/user
        assert config_env['replay_dir'] == '/tmp/user/replay_test/'

    # Test Case 6: Using mock_open for precise control (Alternative approach)
    mock_yaml_data = "key: value"
    with patch("builtins.open", mock_open(read_data=mock_yaml_data)):
        with patch("os.path.exists", return_value=True):
            config_mock = get_config("dummy_path.yaml")
            assert config_mock['key'] == 'value'
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest
import yaml
from pathlib import Path
from unittest.mock import patch, mock_open
from cookiecutter.exceptions import ConfigDoesNotExistException, InvalidConfiguration

def test_get_config(tmp_path):
    # Test case 1: File does not exist
    non_existent_path = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException) as excinfo:
        get_config(non_existent_path)
    assert "does not exist" in str(excinfo.value)

    # Test case 2: Valid YAML file with overrides
    config_data = {
        'cookiecutters_dir': '/tmp/custom_cookies/',
        'abbreviations': {
            'new_abbr': 'https://example.com/{0}'
        }
    }
    config_file = tmp_path / "config.yaml"
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config_data, f)

    config = get_config(config_file)

    # Check if overrides worked
    assert config['cookiecutters_dir'] == os.path.expanduser('/tmp/custom_cookies/')
    # Check if nested merge preserved BUILTIN_ABBREVIATIONS
    assert config['abbreviations']['gh'] == 'https://github.com/{0}.git'
    assert config['abbreviations']['new_abbr'] == 'https://example.com/{0}'
    # Check default values remained for untouched keys
    assert 'replay_dir' in config

    # Test case 3: Invalid YAML syntax
    invalid_yaml_file = tmp_path / "invalid.yaml"
    with open(invalid_yaml_file, 'w', encoding='utf-8') as f:
        f.write("key: : invalid_syntax")
    
    with pytest.raises(InvalidConfiguration) as excinfo:
        get_config(invalid_yaml_file)
    assert "Unable to parse YAML file" in str(excinfo.value)

    # Test case 4: YAML is not a dictionary (e.g., a list or string at top level)
    list_yaml_file = tmp_path / "list.yaml"
    with open(list_yaml_file, 'w', encoding='utf-8') as f:
        f.write("- item1\n- item2")
    
    with pytest.raises(InvalidConfiguration) as excinfo:
        get_config(list_yaml_file)
    assert "should be an object" in str(excinfo.value)

    # Test case 5: Empty YAML file (should return defaults)
    empty_yaml_file = tmp_path / "empty.yaml"
    with open(empty_yaml_file, 'w', encoding='utf-8') as f:
        f.write("")
    
    config_empty = get_config(empty_yaml_file)
    assert config_empty['cookiecutters_dir'] == DEFAULT_CONFIG['cookiecutters_dir']

    # Test case 6: Environment variable expansion in paths
    env_path_file = tmp_path / "env_var.yaml"
    with patch.dict(os.environ, {'TEST_DIR': '/tmp/test_expansion'}):
        with open(env_path_file, 'w', encoding='utf-8') as f:
            f.write("cookiecutters_dir: '$TEST_DIR/cookies/'")
        
        config_env = get_config(env_path_file)
        # Expandvars and expanduser should resolve this to /tmp/test_expansion/cookies/
        assert config_env['cookiecutters_dir'] == os.path.expanduser('/tmp/test_expansion/cookies/')
```


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest
import yaml
from pathlib import Path
from unittest.mock import patch, mock_open
from cookiecutter.exceptions import ConfigDoesNotExistException, InvalidConfiguration

def test_get_config(tmp_path):
    # Test case 1: File does not exist
    non_existent_path = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException) as excinfo:
        get_config(non_existent_path)
    assert "does not exist" in str(excinfo.value)

    # Test case 2: Invalid YAML syntax
    invalid_yaml_path = tmp_path / "invalid.yaml"
    invalid_yaml_path.write_text("key: : value", encoding="utf-8")
    with pytest.raises(InvalidConfiguration) as excinfo:
        get_config(invalid_yaml_path)
    assert "Unable to parse YAML file" in str(excinfo.value)

    # Test case 3: Valid YAML but top-level is not a dictionary (e.g., a list)
    list_yaml_path = tmp_path / "list.yaml"
    list_yaml_path.write_text("- item1\n- item2", encoding="utf-8")
    with pytest.raises(InvalidConfiguration) as excinfo:
        get_config(list_yaml_path)
    assert "should be an object" in str(excinfo.value)

    # Test case 4: Valid configuration and merging
    valid_yaml_path = tmp_path / "valid.yaml"
    custom_content = {
        'replay_dir': '/tmp/custom_replay',
        'abbreviations': {
            'new_abbr': 'https://example.com/{0}'
        },
        'some_new_key': 'some_value'
    }
    valid_yaml_path.write_text(yaml.dump(custom_content), encoding="utf-8")

    config = get_config(valid_yaml_path)

    # Verify merging of top-level keys
    assert config['replay_dir'] == '/tmp/custom_replay'
    assert config['some_new_key'] == 'some_value'
    
    # Verify deep merge of abbreviations (should contain both builtin and new)
    assert 'gh' in config['abbreviations']
    assert config['abbreviations']['new_abbr'] == 'https://example.com/{0}'

    # Verify path expansion for default keys remaining unchanged
    # (Assuming /home/user is not the literal string in the test environment, 
    # but it should be expanded)
    expected_cookies_dir = os.path.expanduser(DEFAULT_CONFIG['cookiecutters_dir'])
    assert config['cookiecutters_dir'] == expected_cookies_dir

    # Test case 5: Mocking open for edge cases with no physical file
    mock_data = "key: value"
    with patch("builtins.open", mock_open(read_data=mock_data)):
        with patch("os.path.exists", return_value=True):
            config = get_config("fake_path.yaml")
            assert config['key'] == 'value'
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest
import yaml
import os
from pathlib import Path
from unittest.mock import patch, mock_open
from cookiecutter.exceptions import ConfigDoesNotExistException, InvalidConfiguration

def test_get_config(tmp_path):
    # Test case 1: File does not exist
    non_existent_path = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException) as excinfo:
        get_config(non_existent_path)
    assert "does not exist" in str(excinfo.value)

    # Test case 2: Valid YAML file
    valid_yaml_content = {
        'cookiecutters_dir': '/tmp/custom_cookies',
        'abbreviations': {'gh': 'https://new-github.com/{0}.git'}
    }
    config_file = tmp_path / "valid_config.yaml"
    config_file.write_text(yaml.dump(valid_yaml_content))

    config = get_config(config_file)
    # Check if merged correctly (abbreviations should have gh updated but keep gl/bb)
    assert config['cookiecutters_dir'] == os.path.expanduser('/tmp/custom_cookies')
    assert config['abbreviations']['gh'] == 'https://new-github.com/{0}.git'
    assert config['abbreviations']['gl'] == 'https://gitlab.com/{0}.git'
    # Check if default values are preserved for keys not in YAML
    assert 'replay_dir' in config

    # Test case 3: Invalid YAML syntax (YAMLError)
    invalid_yaml_file = tmp_path / "invalid_syntax.yaml"
    invalid_yaml_file.write_text("key: [unclosed bracket")
    with pytest.raises(InvalidConfiguration) as excinfo:
        get_config(invalid_yaml_file)
    assert "Unable to parse YAML file" in str(excinfo.value)

    # Test case 4: Top-level element is not a dictionary (e.g., a list)
    list_yaml_file = tmp_path / "list_config.yaml"
    list_yaml_file.write_text("- item1\n- item2")
    with pytest.raises(InvalidConfiguration) as excinfo:
        get_config(list_yaml_file)
    assert "should be an object" in str(excinfo.value)

    # Test case 5: Empty YAML file (should return DEFAULT_CONFIG with expanded paths)
    empty_yaml_file = tmp_path / "empty.yaml"
    empty_yaml_file.write_text("")
    config_empty = get_config(empty_yaml_file)
    assert config_empty['cookiecutters_dir'] == os.path.expanduser(DEFAULT_CONFIG['cookiecutters_dir'])
    assert config_empty['abbreviations']['bb'] == BUILTIN_ABBREVIATIONS['bb']

    # Test case 6: Path expansion (environment variables)
    env_path_content = {'replay_dir': '$HOME/custom_replay'}
    env_config_file = tmp_path / "env_expand.yaml"
    env_config_file.write_text(yaml.dump(env_path_content))
    
    config_expanded = get_config(env_config_file)
    expected_path = os.path.expanduser(os.path.expandvars('$HOME/custom_replay'))
    assert config_expanded['replay_dir'] == expected_path
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest
import yaml
import os
from pathlib import Path
from unittest.mock import patch, mock_open

@pytest.mark.parametrize(
    "config_content, expected_keys, should_raise",
    [
        ({"cookiecutters_dir": "/tmp/custom"}, ["cookiecutters_dir", "replay_dir"], False),
        ({"abbreviations": {"new": "https://new.com/{0}"}}, ["abbreviations"], False),
        ("not a dict", [], True),  # Top-level element is not an object
        ("", [], False),           # Empty file results in empty dict (merged with defaults)
    ],
)
def test_get_config(config_content, expected_keys, should_raise):
    """Test get_config with various valid and invalid YAML contents."""
    yaml_str = yaml.dump(config_content) if isinstance(config_content, dict) else config_content
    path = "/fake/config.yaml"

    # Mocking file existence, open, and os.path.exists
    with patch("os.path.exists", return_value=True), \
         patch("builtins.open", mock_open(read_data=yaml_str)):
        
        if should_raise:
            with pytest.raises(InvalidConfiguration):
                get_config(path)
        else:
            config = get_config(path)
            assert isinstance(config, dict)
            # Check that custom keys exist
            for key in expected_keys:
                if key != "abbreviations": # abbreviations is a complex check
                    assert key in config
            
            # Verify deep merge for abbreviations if we provided one
            if isinstance(config_content, dict) and "abbreviations" in config_content:
                assert "new" in config["abbreviations"]
                assert "gh" in config["abbreviations"]

def test_get_config_file_not_found():
    """Test get_config raises error when file does not exist."""
    with patch("os.path.exists", return_value=False):
        with pytest.raises(ConfigDoesNotExistException):
            get_config("/non/existent/path.yaml")

def test_get_config_invalid_yaml():
    """Test get_config raises error on invalid YAML syntax."""
    invalid_yaml = "key: : value"  # Invalid syntax
    with patch("os.path.exists", return_value=True), \
         patch("builtins.open", mock_open(read_data=invalid_yaml)):
        with pytest.raises(InvalidConfiguration):
            get_config("/fake/config.yaml")

def test_get_config_path_expansion():
    """Test that get_config expands environment variables and user home."""
    config_content = {
        "replay_dir": "$HOME/custom_replay",
        "cookiecutters_dir": "~/custom_cookies"
    }
    yaml_str = yaml.dump(config_content)
    
    with patch("os.path.exists", return_value=True), \
         patch("builtins.open", mock_open(read_data=yaml_str)), \
         patch("os.path.expanduser") as mock_expanduser, \
         patch("os.path.expandvars") as mock_expandvars:
        
        # Setup mocks to simulate expansion
        mock_expandvars.side_effect = lambda x: x.replace("$HOME", "/home/user")
        mock_expanduser.side_effect = lambda x: x.replace("~", "/home/user")
        
        config = get_config("/fake/config.yaml")
        
        assert config["replay_dir"] == "/home/user/custom_replay"
        assert config["cookiecutters_dir"] == "/home/user/custom_cookies"
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest
import yaml
import os
from pathlib import Path
from unittest.mock import patch, mock_open
from cookiecutter.exceptions import ConfigDoesNotExistException, InvalidConfiguration

def test_get_config(tmp_path):
    # Test 1: File does not exist
    non_existent_path = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException) as excinfo:
        get_config(non_existent_path)
    assert "does not exist" in str(excinfo.value)

    # Test 2: Invalid YAML syntax
    invalid_yaml_path = tmp_path / "invalid.yaml"
    invalid_yaml_path.write_text("key: : value", encoding="utf-8")
    with pytest import_error_handling_logic_is_internal_to_get_config:
        # Note: We catch YAMLError inside get_config and raise InvalidConfiguration
        with pytest.raises(InvalidConfiguration) as excinfo:
            get_config(invalid_yaml_path)
        assert "Unable to parse YAML file" in str(excinfo.value)

    # Test 3: YAML is valid but top-level is not a dictionary (e.g., a list)
    list_yaml_path = tmp_path / "list.yaml"
    list_yaml_path.write_text("- item1\n- item2", encoding="utf-8")
    with pytest.raises(InvalidConfiguration) as excinfo:
        get_config(list_yaml_path)
    assert "should be an object" in str(excinfo.value)

    # Test 4: Successful loading and merging with defaults
    valid_yaml_path = tmp_path / "valid.yaml"
    custom_data = {
        'cookiecutters_dir': '/tmp/custom_cookies/',
        'abbreviations': {'new_abbr': 'https://example.com/{0}'}
    }
    valid_yaml_path.write_text(yaml.dump(custom_data), encoding="utf-8")

    config = get_config(valid_yaml_path)

    # Check if custom values are applied
    assert config['cookiecutters_dir'] == os.path.expanduser('/tmp/custom_cookies/')
    # Check if nested dict (abbreviations) was merged correctly
    assert 'new_abbr' in config['abbreviations']
    assert config['abanns'] is not defined # This would fail, verifying we don't overwrite entirely
    assert config['abbreviations']['gh'] == BUILTIN_ABBREVIATIONS['gh']
    # Check if default values are preserved for keys not in the YAML
    assert 'replay_dir' in config
    assert config['replay_dir'] == os.path.expanduser('~/.cookiecutter_replay/')

    # Test 5: Environment variable expansion in paths
    env_path_yaml = tmp_path / "env.yaml"
    env_path_yaml.write_text("cookiecutters_dir: '$HOME/env_test/'", encoding="utf-8")
    config_env = get_config(env_path_yaml)
    assert config_env['cookiecutters_dir'] == os.path.expanduser('$HOME/env_test/').replace('$HOME', os.environ.get('HOME', ''))

def test_get_config_empty_file(tmp_path):
    # Test 6: Empty file should return DEFAULT_CONFIG (merged with empty dict)
    empty_file = tmp_path / "empty.yaml"
    empty_file.write_text("", encoding="utf-8")
    config = get_config(empty_file)
    assert config['cookiecutters_dir'] == DEFAULT_CONFIG['cookiecutters_dir']
```


# LLM-generated content at query #5
#--------------------------

```python
import os
import pytest
from unittest.mock import patch, mock_open

def test_get_user_config():
    # Test Case 1: default_config is True (Return defaults)
    assert get_user_config(default_config=True) == DEFAULT_CONFIG

    # Test Case 2: default_config is a dict (Merge with defaults)
    custom_overrides = {'replay_dir': '/tmp/replay/', 'new_key': 'value'}
    result = get_user_config(default_config=custom_overrides)
    assert result['replay_dir'] == os.path.expanduser('/tmp/replay/')
    assert result['new_key'] == 'value'
    assert result['cookiecutters_dir'] == DEFAULT_CONFIG['cookiecutters_dir']

    # Test Case 3: config_file is provided (Load from custom path)
    yaml_content = "cookiecutters_dir: /custom/path/\nabbreviations:\n  new: 'https://new.com/{0}'"
    with patch("os.path.exists", return_value=True), \
         patch("builtins.open", mock_open(read_data=yaml_content)):
        config = get_user_config(config_file="/custom/config.yaml")
        assert config['cookiecutters_dir'] == os.path.expanduser('/custom/path/')
        assert config['abbreviations']['new'] == 'https://new.com/{0}'
        assert config['abbreviations']['gh'] == BUILTIN_ABBREVIATIONS['gh']

    # Test Case 4: COOKIECUTTER_CONFIG env var is set
    with patch.dict(os.environ, {"COOKIECUTTER_CONFIG": "/env/config.yaml"}), \
         patch("os.path.exists", return_value=True), \
         patch("builtins.open", mock_open(read_data="replay_dir: /env/replay/")):
        config = get_user_config()
        assert config['replay_dir'] == os.path.expanduser('/env/replay/')

    # Test Case 5: No env var and no file exists (Return defaults)
    with patch.dict(os.environ, {}, clear=True), \
         patch("os.path.exists", return_value=False):
        config = get_user_config()
        assert config == DEFAULT_CONFIG

    # Test Case 6: No env var but USER_CONFIG_PATH exists
    with patch.dict(os.environ, {}, clear=True), \
         patch("os.path.exists", side_effect=lambda p: p == USER_CONFIG_PATH), \
         patch("builtins.open", mock_open(read_data="cookiecutters_dir: /user/config/")):
        config = get_user_config()
        assert config['cookiecutters_dir'] == os.path.expanduser('/user/config/')

    # Test Case 7: Invalid YAML in custom config file (Should raise InvalidConfiguration)
    with patch.dict(os.environ, {"COOKIECUTTER_CONFIG": "/invalid.yaml"}), \
         patch("os.path.exists", return_value=True), \
         patch("builtins.open", mock_open(read_data=": invalid yaml")):
        with pytest.raises(InvalidConfiguration):
            get_user_config()

    # Test Case 8: YAML is not a dictionary (Should raise InvalidConfiguration)
    with patch.dict(os.environ, {"COOKIECUTTER_CONFIG": "/list.yaml"}), \
         patch("os.path.exists", return_value=True), \
         patch("builtins.open", mock_open(read_data="- item1\n- item2")):
        with pytest.raises(InvalidConfiguration):
            get_user_config()

    # Test Case 9: File does not exist (Should raise ConfigDoesNotExistException)
    with patch.dict(os.environ, {"COOKIECUTTER_CONFIG": "/nonexistent.yaml"}), \
         patch("os.path.exists", return_value=False):
        with pytest.raises(ConfigDoesNotExistException):
            get_user_config()
```


# LLM-generated content at query #6
#--------------------------

```python
import os
import pytest
from unittest.mock import patch, mock_open

def test_get_user_config():
    # Test Case 1: default_config is True (Returns DEFAULT_CONFIG copy)
    config = get_user_config(default_config=True)
    assert config == DEFAULT_CONFIG
    assert config is not DEFAULT_CONFIG

    # Test Case 2: default_config is a dict (Merges provided dict with DEFAULT_CONFIG)
    custom_values = {'replay_dir': '/tmp/replay', 'new_key': 'value'}
    config = get_unmodified_user_config_helper(default_config=custom_values)
    assert config['replay_dir'] == os.path.expanduser('/tmp/replay')
    assert config['new_key'] == 'value'
    assert config['cookiecutters_dir'] == DEFAULT_CONFIG['cookiecutters_dir']

    # Test Case 3: config_file provided (Loads from specific path)
    yaml_content = "abbreviations:\n  custom: 'https://custom.com/{0}'\nreplay_dir: '$HOME/test_replay'"
    with patch("os.path.exists", return_value=True), \
         patch("builtins.open", mock_open(read_data=yaml_content)):
        config = get_user_config(config_file="/fake/path/config.yaml")
        assert config['abbreviations']['custom'] == 'https://custom.com/{0}'
        assert config['abbreviations']['gh'] == BUILTIN_ABBREVIATIONS['gh']
        assert config['replay_dir'] == os.path.expanduser('$HOME/test_replay')

    # Test Case 4: COOKIECUTTER_CONFIG environment variable is set
    yaml_content = "cookiecutters_dir: '/custom/dir'"
    with patch.dict(os.environ, {"COOKIECUTTER_CONFIG": "/env/config.yaml"}), \
         patch("os.path.exists", return_value=True), \
         patch("builtins.open", mock_open(read_data=yaml_content)):
        config = get_user_config()
        assert config['cookiecutters_dir'] == os.path.expanduser('/custom/dir')

    # Test Case 5: No env var, no custom file, and USER_CONFIG_PATH does not exist (Returns DEFAULT_CONFIG)
    with patch.dict(os.environ, {}, clear=True), \
         patch("os.path.exists", return_value=False):
        config = get_user_config()
        assert config == DEFAULT_CONFIG

    # Test Case 6: No env var, no custom file, but USER_CONFIG_PATH exists (Loads from ~/.cookiecutterrc)
    yaml_content = "replay_dir: '/user/replay'"
    with patch.dict(os.environ, {}, clear=True), \
         patch("os.path.exists", side_effect=lambda p: p == USER_CONFIG_PATH), \
         patch("builtins.open", mock_open(read_data=yaml_content)):
        config = get_user_config()
        assert config['replay_dir'] == os.path.expanduser('/user/replay')

def get_unmodified_user_config_helper(default_config):
    """Helper to bypass environment/file logic for simple dict merge tests."""
    return get_user_config(default_config=default_config)
```


# LLM-generated content at query #7
#--------------------------

```python
import os
import pytest
from unittest.mock import patch, mock_open

def test_get_user_config(tmp_path):
    # Test Case 1: default_config is True (Return defaults)
    config = get_user_config(default_config=True)
    assert config == DEFAULT_CONFIG
    assert config is not DEFAULT_CONFIG  # Should be a copy

    # Test Case 2: default_config is a dict (Merge with defaults)
    custom_values = {'replay_dir': '/tmp/replay', 'new_key': 'value'}
    config = get_user_config(default_config=custom_values)
    assert config['replay_dir'] == os.path.expanduser('/tmp/replay')
    assert config['new_key'] == 'value'
    assert config['abbreviations'] == BUILTIN_ABBREVIATIONS

    # Test Case 3: custom config_file provided and exists
    custom_config_file = tmp_path / "custom_config.yaml"
    custom_content = "cookiecutters_dir: '/tmp/custom_dir'\nabbreviations:\n  new: 'https://new.com/{0}'"
    custom_config_file.write_text(custom_content)
    
    config = get_user_config(config_file=str(custom_config_file))
    assert config['cookiecutters_dir'] == os.path.expanduser('/tmp/custom_dir')
    assert config['abbreviations']['new'] == 'https://new.com/{0}'
    assert config['abbreviations']['gh'] == BUILTIN_ABBREVIATIONS['gh']

    # Test Case 4: custom config_file provided but does not exist
    with pytest.raises(ConfigDoesNotExistException):
        get_user_config(config_file="non_existent_path.yaml")

    # Test Case 5: Using COOKIECUTTER_CONFIG environment variable
    env_config_file = tmp_path / "env_config.yaml"
    env_content = "replay_dir: '/tmp/env_replay'"
    env_config_file.write_text(env_content)

    with patch.dict(os.environ, {'COOKIECUTTER_CONFIG': str(env_config_file)}):
        config = get_user_config()
        assert config['replay_dir'] == os.path.expanduser('/tmp/env_replay')

    # Test Case 6: Fallback to USER_CONFIG_PATH when env var is not set
    # We mock os.path.exists for USER_CONFIG_PATH and the file content
    with patch('os.environ', {}), \
         patch('os.path.exists') as mock_exists, \
         patch('builtins.open', mock_open(read_data="cookiecutters_dir: '/tmp/user_dir'")):
        
        # Setup: USER_CONFIG_PATH exists, but other paths (like the one in code) don't
        def side_effect(path):
            if path == USER_CONFIG_PATH:
                return True
            return False
        
        mock_exists.side_effect = side_effect
        
        config = get_user_config()
        assert config['cookiecutters_dir'] == os.path.expanduser('/tmp/user_dir')

    # Test Case 7: Fallback to DEFAULT_CONFIG when no env var and no USER_CONFIG_PATH exists
    with patch('os.environ', {}), \
         patch('os.path.exists', return_value=False):
        config = get_user_config()
        assert config == DEFAULT_CONFIG

    # Test Case 8: Invalid YAML content in config file
    invalid_yaml_file = tmp_path / "invalid.yaml"
    invalid_yaml_file.write_text("key: : value") # Broken YAML
    with pytest.raises(InvalidConfiguration):
        get_user_config(config_file=str(invalid_yaml_file))

    # Test Case 9: Top-level element is not a dictionary
    not_a_dict_file = tmp_path / "not_a_dict.yaml"
    not_a_dict_file.write_text("- item1\n- item2") # YAML list, not dict
    with pytest.raises(InvalidConfiguration, match="Top-level element of YAML file"):
        get_user_config(config_file=str(not_a_dict_file))
```


