####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest
import yaml
import os
from pathlib import Path
from unittest.mock import patch, mock_open

def test_get_config(tmp_path):
    # Test 1: Config file does not exist
    non_existent_path = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException) as excinfo:
        get_config(non_existing_path)
    assert "does not exist" in str(excinfo.value)

    # Test 2: Config file contains invalid YAML
    invalid_yaml_path = tmp_path / "invalid.yaml"
    invalid_yaml_path.write_text("key: : value", encoding="utf-8")
    with pytest.raises(InvalidConfiguration) as excinfo:
        get_config(invalid_yaml_path)
    assert "Unable to parse YAML file" in str(excinfo.value)

    # Test 3: Config file contains top-level list instead of dict
    list_yaml_path = tmp_path / "list.yaml"
    list_yaml_path.write_text("- item1\n- item2", encoding="utf-8")
    with pytest.raises(InvalidConfiguration) as excinfo:
        get_config(list_yaml_path)
    assert "should be an object" in str(excinfo.value)

    # Test 4: Valid config file with overrides and path expansion
    # We use a specific env var to test _expand_path logic
    os.environ["TEST_VAR"] = "test_dir"
    valid_yaml_content = {
        "cookiecutters_dir": "$TEST_VAR/cookies",
        "abbreviations": {
            "new_abbr": "https://example.com/{0}"
        },
        "custom_key": "custom_value"
    }
    
    valid_config_path = tmp_path / "valid.yaml"
    with open(valid_config_path, "w", encoding="utf-8") as f:
        yaml.dump(valid_yaml_content, f)

    config = get_config(valid_config_path)

    # Verify merged abbreviations (preserved 'gh', 'gl', 'bb' and added 'new_abbr')
    assert "gh" in config["abbreviations"]
    assert config["abbreviations"]["new_abbr"] == "https://example.com/{0}"
    
    # Verify path expansion worked
    assert config["cookiecutters_dir"] == os.path.expanduser("test_dir/cookies")
    
    # Verify custom key exists
    assert config["custom_key"] == "custom_value"
    
    # Verify default value remains if not overwritten
    assert "replay_dir" in config
    
    # Cleanup env var
    del os.environ["TEST_VAR"]
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
    # Test Case 1: Config file does not exist
    non_existent_path = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException) as excinfo:
        get_config(non_existent_path)
    assert "does not exist" in str(excinfo.value)

    # Test Case 2: Config file is valid YAML
    valid_config_content = {
        'cookiecutters_dir': '~/custom_cookiecutters/',
        'abbreviations': {
            'custom': 'https://custom.com/{0}'
        }
    }
    config_file = tmp_path / "valid_config.yaml"
    config_file.write_text(yaml.dump(valid_config_content), encoding='utf-8')
    
    config = get_config(config_file)
    
    # Verify merged content
    assert config['abbreviations']['custom'] == 'https://custom.com/{0}'
    assert config['abbreviations']['gh'] == 'https://github.com/{0}.git'
    # Verify path expansion (assuming home is expanded)
    assert config['cookiecutters_dir'].endswith('custom_cookiecutters/')
    assert os.path.expanduser('~') in config['cookiecutters_dir']

    # Test Case 3: Config file contains invalid YAML syntax
    invalid_yaml_file = tmp_path / "invalid_syntax.yaml"
    invalid_yaml_file.write_text("key: : value", encoding='utf-8')
    with pytest.raises(InvalidConfiguration) as excinfo:
        get_config(invalid_yaml_file)
    assert "Unable to parse YAML file" in str(excinfo.value)

    # Test Case 4: Config file is valid YAML but not a dictionary (top-level list)
    list_yaml_file = tmp_path / "list_config.yaml"
    list_yaml_file.write_text("- item1\n- item2", encoding='utf-8')
    with pytest.raises(InvalidConfiguration) as excinfo:
        get_config(list_yaml_file)
    assert "should be an object" in str(excinfo.value)

    # Test Case 5: Config file is empty
    empty_yaml_file = tmp_path / "empty.yaml"
    empty_yaml_file.write_text("", encoding='utf-8')
    config_empty = get_config(empty_yaml_file)
    # Should return DEFAULT_CONFIG
    assert config_empty['replay_dir'] == os.path.expanduser('~/.cookiecutter_replay/')
    assert config_empty['abbreviations']['bb'] == 'https://bitbucket.org/{0}'

    # Test Case 6: Mocking open for specific content injection
    mock_data = "replay_dir: $ENV_VAR_PATH"
    with patch("builtins.open", mock_open(read_data=mock_data)):
        with patch("os.path.exists", return_value=True):
            with patch.dict(os.environ, {"ENV_VAR_PATH": "/tmp/replay"}):
                config_mock = get_config(config_file)
                assert config_mod_path := config_mock['replay_dir']
                assert config_mod_path.endswith("/tmp/replay")
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest
import yaml
from pathlib import Path
from unittest.mock import patch, mock_open

def test_get_config(tmp_path):
    # Test case 1: Config file does not exist
    non_existent_path = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException) as excinfo:
        get_config(non_existent_path)
    assert "does not exist" in str(excinfo.value)

    # Test case 2: Valid config file with overrides
    config_data = {
        'cookiecutters_dir': '~/custom_templates',
        'abbreviations': {
            'custom': 'https://custom.com/{0}'
        }
    }
    config_file = tmp_path / "valid_config.yaml"
    config_file.write_text(yaml.dump(config_data))

    config = get_config(config_file)
    
    # Verify overrides and path expansion
    assert config['cookiecutters_dir'] == os.path.expanduser('~/custom_templates')
    # Verify deep merge of abbreviations
    assert config['abbreviations']['gh'] == 'https://github.com/{0}.git'
    assert config['abbreviations']['custom'] == 'https://custom.com/{0}'
    # Verify defaults are preserved
    assert 'replay_dir' in config

    # Test case 3: Invalid YAML syntax
    invalid_yaml_file = tmp_path / "invalid.yaml"
    invalid_yaml_file.write_text("key: : value :") # Broken YAML
    with pytest.raises(InvalidConfiguration) as excinfo:
        get_config(invalid_yaml_file)
    assert "Unable to parse YAML file" in str(excinfo.value)

    # Test case 4: YAML is not a dictionary (top-level list)
    list_yaml_file = tmp_path / "list.yaml"
    list_yaml_file.write_text("- item1\n- item2")
    with pytest.mock.patch("builtins.open", mock_open(read_data="- item1")):
        # We need to ensure the file exists for the os.path.exists check
        with pytest.raises(InvalidConfiguration) as excinfo:
            get_config(list_yaml_file)
        assert "should be an object" in str(excinfo.value)

    # Test case 5: Empty YAML file (should return defaults)
    empty_yaml_file = tmp_path / "empty.yaml"
    empty_yaml_file.write_text("")
    config_empty = get_config(empty_yaml_file)
    assert config_empty['cookiecutters_dir'] == DEFAULT_CONFIG['cookiecutters_dir']

    # Test case 6: Environment variable expansion in config
    env_var_config = tmp_path / "env_var.yaml"
    env_var_config.write_text({'replay_dir': '$HOME/replay_test'})
    
    with patch.dict(os.environ, {'HOME': '/tmp/user'}):
        config_env = get_config(env_var_config)
        assert config_env['replay_dir'] == '/tmp/user/replay_test'
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest
import yaml
import os
from pathlib import Path
from unittest.mock import patch, mock_open

def test_get_config(tmp_path):
    # Test 1: File does not exist
    non_existent_path = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException) as excinfo:
        get_config(non_existent_path)
    assert "does not exist" in str(excinfo.value)

    # Test 2: Invalid YAML syntax
    invalid_yaml_path = tmp_path / "invalid.yaml"
    invalid_yaml_path.write_text("key: [unclosed bracket", encoding="utf-8")
    with pytest.raises(InvalidConfiguration) as excinfo:
        get_config(invalid_yaml_path)
    assert "Unable to parse YAML file" in str(excinfo.value)

    # Test 3: YAML is not a dictionary (top-level is a list)
    list_yaml_path = tmp_path / "list.yaml"
    list_yaml_path.write_text("- item1\n- item2", encoding="utf-8")
    with pytest.py.raises(InvalidConfiguration) as excinfo:
        get_config(list_yaml_path)
    assert "should be an object" in str(excinfo.value)

    # Test 4: Valid config loading and merging
    valid_yaml_path = tmp_path / "valid.yaml"
    custom_content = {
        'replay_dir': '~/custom_replay',
        'abbreviations': {
            'new_abbr': 'https://example.com/{0}'
        },
        'new_key': 'new_value'
    }
    yaml.dump(custom_content, valid_yaml_path)

    config = get_config(valid_yaml_path)

    # Check merged keys
    assert config['new_key'] == 'new_value'
    # Check path expansion
    assert config['replay_dir'] == os.path.expanduser('~/custom_replay')
    # Check nested dict merge (abbreviations should contain both built-in and new)
    assert 'gh' in config['abbreviations']
    assert config['abbreviations']['new_abbr'] == 'https://example.com/{0}'
    # Check default preservation
    assert config['cookiecutters_dir'] == os.path.expanduser('~/.cookiecutters/')

    # Test 5: Empty YAML file
    empty_yaml_path = tmp_path / "empty.yaml"
    empty_yaml_path.write_text("", encoding="utf-8")
    config_empty = get_config(empty_yaml_path)
    # Should return DEFAULT_CONFIG with expanded paths
    assert config_empty['cookiecutters_dir'] == os.path.expanduser('~/.cookiecutters/')
    assert config_empty['abbreviations']['gh'] == 'https://github.com/{0}.git'
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest
import yaml
import os
from pathlib import Path

def test_get_config(tmp_path):
    # Test 1: File does not exist
    non_existent_path = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException) as excinfo:
        get_config(non_existent_path)
    assert "does not exist" in str(excinfo.value)

    # Test 2: Invalid YAML syntax
    invalid_yaml_path = tmp_path / "invalid.yaml"
    invalid_yaml_path.write_text("key: : value", encoding="utf-8")
    with pytest.raises(InvalidConfiguration) as excinfo:
        get_config(invalid_yaml_path)
    assert "Unable to parse YAML file" in str(excinfo.value)

    # Test 3: YAML is not a dictionary (top-level element is a list)
    list_yaml_path = tmp_path / "list.yaml"
    list_yaml_path.write_text("- item1\n- item2", encoding="utf-8")
    with pytest.raises(InvalidConfiguration) as excinfo:
        get_config(list_yaml_path)
    assert "should be an object" in str(excinfo.value)

    # Test 4: Valid config with overrides and path expansion
    # We use an environment variable in the config to test _expand_path
    valid_yaml_path = tmp_path / "valid.yaml"
    config_data = {
        'cookiecutters_dir': '$TEST_VAR/cookies',
        'abbreviations': {
            'custom': 'https://custom.com/{0}'
        },
        'replay_dir': '~/custom_replay'
    }
    valid_yaml_path.write_text(yaml.dump(config_data), encoding="utf-8")
    
    # Mock environment variable for expansion test
    os.environ['TEST_VAR'] = '/tmp/test_dir'
    
    try:
        config = get_config(valid_yaml_path)
        
        # Verify merge logic: defaults preserved where not overwritten
        assert config['abbreviations']['gh'] == BUILTIN_ABBREVIATIONS['gh']
        # Verify merge logic: nested dicts updated
        assert config['abbreviations']['custom'] == 'https://custom.com/{0}'
        
        # Verify path expansion (env var)
        assert config['cookiecutters_dir'] == '/tmp/test_dir/cookies'
        
        # Verify path expansion (user home)
        expected_home_replay = os.path.expanduser('~/custom_replay')
        assert config['replay_dir'] == expected_home_replay
        
        # Verify value override
        assert config['replay_dir'] != DEFAULT_CONFIG['replay_dir']
        
    finally:
        if 'TEST_VAR' in os.environ:
            del os.environ['TEST_VAR']

    # Test 5: Empty YAML file (should return DEFAULT_CONFIG with expanded paths)
    empty_yaml_path = tmp_path / "empty.yaml"
    empty_yaml_path.write_text("", encoding="utf-8")
    config_empty = get_config(empty_yaml_path)
    assert config_empty['cookiecutters_dir'] == _expand_path(DEFAULT_CONFIG['cookiecutters_dir'])
    assert config_empty['abbreviations'] == BUILTIN_ABBREVIATIONS
```


# LLM-generated content at query #6
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

    # Test 2: Valid YAML file
    config_data = {
        'cookiecutters_dir': '~/custom_cookies/',
        'abbreviations': {
            'gh': 'https://custom-github.com/{0}.git'
        }
    }
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump(config_data), encoding='utf-8')
    
    config = get_config(config_file)
    
    # Check if defaults are merged correctly
    assert config['replay_dir'] == os.path.expanduser('~/.cookiecutter_replay/')
    # Check if custom value is applied and path expanded
    assert config['cookiecutters_dir'] == os.path.expanduser('~/custom_cookies/')
    # Check if nested dicts are merged (abbreviations)
    assert config['abbreviations']['gh'] == 'https://custom-github.com/{0}.git'
    assert config['abbreviations']['gl'] == 'https://gitlab.com/{0}.git'

    # Test 3: Invalid YAML syntax
    invalid_yaml_file = tmp_path / "invalid.yaml"
    invalid_yaml_file.write_text("key: : value", encoding='utf-8') # Invalid YAML
    with pytest.raises(InvalidConfiguration) as excinfo:
        get_config(invalid_yaml_file)
    assert "Unable to parse YAML file" in str(excinfo.value)

    # Test 4: YAML is not a dictionary (top-level element is a list)
    list_yaml_file = tmp_path / "list.yaml"
    list_yaml_file.write_text("- item1\n- item2", encoding='utf-8')
    with pytest.raises(InvalidConfiguration) as excinfo:
        get_config(list_yaml_file)
    assert "should be an object" in str(excinfo.value)

    # Test 5: Empty YAML file (should return defaults)
    empty_yaml_file = tmp_path / "empty.yaml"
    empty_yaml_file.write_text("", encoding='utf-8')
    config = get_config(empty_yaml_file)
    assert config['cookiecutters_dir'] == os.path.expanduser('~/.cookiecutters/')
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest
import os
from unittest.mock import patch, mock_open

def test_get_user_config():
    # Test Case 1: default_config is True (returns defaults)
    config = get_user_config(default_config=True)
    assert config == DEFAULT_CONFIG
    assert config is not DEFAULT_CONFIG  # Should be a copy

    # Test Case 2: default_config is a dict (merges with defaults)
    custom_values = {'replay_dir': '/tmp/replay', 'new_key': 'value'}
    config = get_user_config(default_config=custom_values)
    assert config['replay_dir'] == os.path.expanduser('/tmp/replay')
    assert config['new_key'] == 'value'
    assert config['cookiecutters_dir'] == DEFAULT_CONFIG['cookiecutters_dir']

    # Test Case 3: config_file provided (loads custom path)
    custom_path = "/tmp/custom_config.yaml"
    yaml_content = "abbreviations:\n  new: 'https://new.com/{0}'\nreplay_dir: '/tmp/custom_replay'"
    
    with patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data=yaml_content)):
        config = get_user_config(config_file=custom_path)
        assert config['abbreviations']['new'] == 'https://new.com/{0}'
        assert config['abbreviations']['gh'] == BUILTIN_ABBREVIATIONS['gh']
        assert config['replay_dir'] == os.path.expanduser('/tmp/custom_replay')

    # Test Case 4: Using COOKIECUTTER_CONFIG environment variable
    env_path = "/tmp/env_config.yaml"
    env_yaml = "cookiecutters_dir: '/tmp/env_cookies'"
    
    with patch.dict(os.environ, {'COOKIECUTTER_CONFIG': env_path}), \
         patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data=env_yaml)):
        config = get_user_config()
        assert config['cookiecutters_dir'] == os.path.expanduser('/tmp/env_cookies')

    # Test Case 5: Fallback to USER_CONFIG_PATH when env var is not set
    with patch.dict(os.environ, {}, clear=True), \
         patch('os.path.exists') as mock_exists, \
         patch('builtins.open', mock_open(read_data="replay_dir: '/tmp/user_replay'")):
        
        # First call to exists returns True for USER_CONFIG_PATH
        mock_exists.side_effect = lambda p: p == USER_CONFIG_PATH
        
        config = get_user_config()
        assert config['replay_dir'] == os.path.expanduser('/tmp/user_replay')

    # Test Case 6: Fallback to DEFAULT_CONFIG when nothing is found
    with patch.dict(os.environ, {}, clear=True), \
         patch('os.path.exists', return_value=False):
        config = get_user_config()
        assert config == DEFAULT_CONFIG

    # Test Case 7: Error when COOKIECUTTER_CONFIG points to non-existent file
    with patch.dict(os.environ, {'COOKIECUTTER_CONFIG': '/non/existent/path.yaml'}), \
         patch('os.path.exists', return_value=False):
        with pytest.raises(ConfigDoesNotExistException):
            get_user_config()

    # Test Case 8: Error when YAML is invalid
    invalid_yaml = "key: : : invalid"
    with patch.dict(os.environ, {'COOKIECUTTER_CONFIG': '/tmp/bad.yaml'}), \
         patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data=invalid_yaml)):
        with pytest.raises(InvalidConfiguration):
            get_user_config()
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest
import yaml
import os
from pathlib import Path
from unittest.mock import patch, mock_open
from cookiecutter.exceptions import ConfigDoesNotExistException, InvalidConfiguration

def test_get_config(tmp_path):
    # Test 1: Config file does not exist
    non_existent_path = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException) as excinfo:
        get_config(non_existent_path)
    assert "does not exist" in str(excinfo.value)

    # Test 2: Valid config file with overrides
    config_data = {
        'cookiecutters_dir': '/tmp/custom_cookiecutters',
        'abbreviations': {
            'gh': 'https://custom-github.com/{0}.git',
            'new_key': 'value'
        }
    }
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump(config_data), encoding='utf-8')

    config = get_config(config_file)

    # Check that defaults are preserved
    assert config['replay_dir'] == os.path.expanduser('~/.cookiecutter_replay/')
    # Check that overrides work
    assert config['cookiecutters_dir'] == '/tmp/custom_cookiecutters'
    # Check that nested dict merge works (abbreviations)
    assert config['abbreviations']['gh'] == 'https://custom-github.com/{0}.git'
    assert config['abbreviations']['bb'] == 'https://bitbucket.org/{0}'
    assert config['abbreviations']['new_key'] == 'value'

    # Test 3: Invalid YAML syntax
    invalid_yaml_file = tmp_path / "invalid.yaml"
    invalid_yaml_file.write_text("key: : value", encoding='utf-8')
    with pytest.raises(InvalidConfiguration) as excinfo:
        get_config(invalid_yaml_file)
    assert "Unable to parse YAML" in str(excinfo.value)

    # Test 4: YAML is not a dictionary (top-level list)
    list_yaml_file = tmp_path / "list.yaml"
    list_yaml_file.write_text("- item1\n- item2", encoding='utf-8')
    with pytest.raises(InvalidConfiguration) as excinfo:
        get_config(list_yaml_file)
    assert "should be an object" in str(excinfo.value)

    # Test 5: Empty YAML file (should return defaults)
    empty_yaml_file = tmp_path / "empty.yaml"
    empty_yaml_file.write_text("", encoding='utf-8')
    config_empty = get_config(empty_yaml_file)
    assert config_empty['cookiecutters_dir'] == os.path.expanduser('~/.cookiecutters/')

    # Test 6: Path expansion in config
    env_path_config = tmp_path / "env_path.yaml"
    env_path_config.write_text("cookiecutters_dir: $HOME/custom_dir", encoding='utf-8')
    config_expanded = get_config(env_path_config)
    assert config_expanded['cookiecutters_dir'] == os.path.expanduser('~/custom_dir')
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest
import yaml
from pathlib import Path
from unittest.mock import patch, mock_open

def test_get_config(tmp_path):
    # Test 1: Config file does not exist
    non_existent_path = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException) as excinfo:
        get_config(non_existent_path)
    assert "does not exist" in str(excinfo.value)

    # Test 2: Config file has invalid YAML syntax
    invalid_yaml_path = tmp_path / "invalid.yaml"
    invalid_yaml_path.write_text("key: : value", encoding="utf-8")
    with pytest.raises(InvalidConfiguration) as excinfo:
        get_config(invalid_yaml_path)
    assert "Unable to parse YAML file" in str(excinfo.value)

    # Test 3: Config file has top-level list instead of dict
    list_yaml_path = tmp_path / "list.yaml"
    list_yaml_path.write_text("- item1\n- item2", encoding="utf-8")
    with pytest.raises(InvalidConfiguration) as excinfo:
        get_config(list_yaml_path)
    assert "should be an object" in str(excinfo.value)

    # Test 4: Valid config file with overrides and path expansion
    # We mock os.path.expanduser to ensure deterministic results for the test
    valid_yaml_content = {
        'cookiecutters_dir': '~/custom_templates',
        'abbreviations': {
            'gh': 'https://custom-github.com/{0}.git',
            'new_key': 'new_value'
        },
        'custom_key': 'custom_value'
    }
    valid_config_path = tmp_path / "valid.yaml"
    with open(valid_config_path, 'w', encoding='utf-8') as f:
        yaml.dump(valid_yaml_content, f)

    with patch('os.path.expanduser', side_effect=lambda x: x.replace('~', '/tmp/home')):
        config = get_config(valid_config_path)

        # Check if default values are merged with overrides
        assert config['custom_key'] == 'custom_value'
        
        # Check if nested dict (abbreviations) is merged, not overwritten
        assert config['abbreviations']['gh'] == 'https://custom-github.com/{0}.git'
        assert config['abbreviations']['bb'] == 'https://bitbucket.org/{0}'
        assert config['abbreviations']['new_key'] == 'new_value'

        # Check if path expansion worked
        assert config['cookiecutters_dir'] == '/tmp/home/custom_templates'
        
        # Check if default replay_dir was preserved and expanded
        assert config['replay_dir'] == '/tmp/home/.cookiecutter_replay/'

    # Test 5: Empty YAML file returns default config
    empty_yaml_path = tmp_path / "empty.yaml"
    empty_yaml_path.write_text("", encoding="utf-8")
    with patch('os.path.expanduser', side_effect=lambda x: x.replace('~', '/tmp/home')):
        config_empty = get_config(empty_yaml_path)
        assert config_empty['cookiecutters_dir'] == '/tmp/home/.cookiecutters/'
        assert config_empty['abbreviations']['gh'] == 'https://github.com/{0}.git'
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest
import yaml
from pathlib import Path
from unittest.mock import patch, mock_open

def test_get_config(tmp_path):
    """Test get_config with various scenarios: valid file, invalid YAML, non-dict YAML, and missing file."""
    
    # 1. Test: Config file does not exist
    non_existent_path = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException) as excinfo:
        get_config(non_existent_path)
    assert "does not exist" in str(excinfo.value)

    # 2. Test: Valid configuration file
    valid_config_content = {
        'cookiecutters_dir': '~/custom_templates/',
        'abbreviations': {
            'gh': 'https://github.com/custom/{0}.git',
            'new_key': 'value'
        }
    }
    valid_config_path = tmp_path / "valid_config.yaml"
    with open(valid_config_path, 'w', encoding='utf-8') as f:
        yaml.dump(valid_config_content, f)

    config = get_config(valid_config_path)
    
    # Check if merged correctly
    assert config['cookiecutters_dir'] == os.path.expanduser('~/custom_templates/')
    # Check if abbreviations preserved existing keys and added new ones
    assert config['abbreviations']['gh'] == 'https://github.com/custom/{0}.git'
    assert config['abbreviations']['bb'] == 'https://bitbucket.org/{0}'
    assert config['abbreviations']['new_key'] == 'value'
    # Check if default replay_dir is still present
    assert 'replay_dir' in config

    # 3. Test: Invalid YAML syntax
    invalid_yaml_path = tmp_path / "invalid_syntax.yaml"
    with open(invalid_yaml_path, 'w', encoding='utf-8') as f:
        f.write("key: [unclosed_bracket")
    
    with pytest.raises(InvalidConfiguration) as excinfo:
        get_config(invalid_yaml_path)
    assert "Unable to parse YAML file" in str(excinfo.value)

    # 4. Test: YAML is not a dictionary (top-level is a list)
    list_yaml_path = tmp_path / "list_config.yaml"
    with open(list_yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(['item1', 'item2'], f)
    
    with pytest.raises(InvalidConfiguration) as excinfo:
        get_config(list_yaml_path)
    assert "should be an object" in str(excinfo.value)

    # 5. Test: Empty YAML file (should return defaults)
    empty_yaml_path = tmp_path / "empty_config.yaml"
    with open(empty_yaml_path, 'w', encoding='utf-8') as f:
        f.write("")
    
    empty_config = get_config(empty_yaml_path)
    assert empty_config['cookiecutters_dir'] == DEFAULT_CONFIG['cookiecutters_dir']
    assert empty_config['abbreviations'] == BUILTIN_ABBREVIATIONS

    # 6. Test: Environment variable expansion in paths
    env_var_path = tmp_path / "env_config.yaml"
    with open(env_var_path, 'w', encoding='utf-8') as f:
        yaml.dump({'cookiecutters_dir': '$HOME/env_test/'}, f)
    
    with patch.dict(os.environ, {'HOME': '/tmp/user'}):
        config_env = get_config(env_var_path)
        assert config_env['cookiecutters_dir'] == '/tmp/user/env_test/'
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest
import yaml
import os
from pathlib import Path
from unittest.mock import patch, mock_open

def test_get_config(tmp_path):
    # Test case 1: Config file does not exist
    non_existent_path = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException) as excinfo:
        get_config(non_existent_path)
    assert "does not exist" in str(excinfo.value)

    # Test case 2: Invalid YAML syntax
    invalid_yaml_path = tmp_path / "invalid.yaml"
    invalid_yaml_path.write_text("key: [unclosed bracket", encoding="utf-8")
    with pytest.raises(InvalidConfiguration) as excinfo:
        get_config(invalid_yaml_path)
    assert "Unable to parse YAML file" in str(excinfo.value)

    # Test case 3: YAML is not a dictionary (top-level list)
    list_yaml_path = tmp_path / "list.yaml"
    list_yaml_path.write_text("- item1\n- item2", encoding="utf-8")
    with pytest.raises(InvalidConfiguration) as excinfo:
        get_config(list_yaml_path)
    assert "should be an object" in str(excinfo.value)

    # Test case 4: Valid config with overrides and path expansion
    valid_yaml_path = tmp_path / "valid.yaml"
    custom_config_content = {
        'cookiecutters_dir': '$HOME/custom_cookies',
        'abbreviations': {
            'custom': 'https://example.com/{0}'
        },
        'some_new_key': 'some_value'
    }
    with open(valid_yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(custom_config_content, f)

    # We mock os.path.expanduser to ensure deterministic results for the test
    # regardless of the host machine's environment
    with patch('os.path.expanduser', side_effect=lambda x: x.replace('$HOME', '/tmp/mock_home')):
        config = get_config(valid_yaml_path)

        # Check that default values are preserved
        assert config['replay_dir'] == os.path.expanduser('~/.cookiecutter_replay/')
        
        # Check that overrides worked
        assert config['some_new_key'] == 'some_value'
        
        # Check that nested dicts (abbreviations) were merged, not overwritten
        assert 'gh' in config['abbreviations']
        assert config['abbreviations']['custom'] == 'https://example.com/{0}'
        
        # Check that path expansion worked for the overridden path
        # The expansion logic in get_config calls _expand_path
        assert config['cookiecutters_dir'].startswith('/tmp/mock_home')
        assert 'custom_cookies' in config['cookiecutters_dir']

    # Test case 5: Empty YAML file (should return defaults)
    empty_yaml_path = tmp_path / "empty.yaml"
    empty_yaml_path.write_text("", encoding="utf-8")
    config_empty = get_config(empty_yaml_path)
    assert config_empty['cookiecutters_dir'] == DEFAULT_CONFIG['cookiecutters_dir']
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

def test_get_user_config():
    # Test 1: default_config=True (Return DEFAULT_CONFIG)
    config = get_user_config(default_config=True)
    assert config == DEFAULT_CONFIG
    assert config is not DEFAULT_CONFIG  # Should be a copy

    # Test 2: default_config is a dict (Merge with DEFAULT_CONFIG)
    custom_overrides = {'replay_dir': '/tmp/custom_replay/', 'new_key': 'new_val'}
    config = get_user_config(default_config=custom_overrides)
    assert config['replay_dir'] == os.path.expanduser('/tmp/custom_replay/')
    assert config['new_key'] == 'new_val'
    assert config['cookiecutters_dir'] == DEFAULT_CONFIG['cookiecutters_dir']

    # Test 3: config_file provided (Custom path)
    # We mock get_config to avoid actual file I/O
    mock_config_data = {'cookiecutters_dir': '/custom/path/'}
    with patch('__main__.get_config', return_value=mock_config_data) as mock_get:
        config = get_user_config(config_file='/tmp/custom_config.yaml')
        mock_get.assert_called_once_with('/tmp/custom_config.yaml')
        assert config['cookiecutters_dir'] == '/custom/path/'

    # Test 4: COOKIECUTTER_CONFIG env var is set
    with patch.dict(os.environ, {'COOKIECUTTER_CONFIG': '/env/path/config.yaml'}):
        with patch('__main__.get_config', return_value=DEFAULT_CONFIG) as mock_get:
            config = get_user_config()
            mock_get.assert_called_once_with('/env/path/config.yaml')
            assert config == DEFAULT_CONFIG

    # Test 5: COOKIECUTTER_CONFIG not set, USER_CONFIG_PATH exists
    with patch.dict(os.environ, {}, clear=True):
        with patch('os.path.exists', return_value=True):
            with patch('__main__.get_config', return_value=DEFAULT_CONFIG) as mock_get:
                config = get_user_config()
                # Should look for USER_CONFIG_PATH
                mock_get.assert_called_once_with(USER_CONFIG_PATH)
                assert config == DEFAULT_CONFIG

    # Test 6: COOKIECUTTER_CONFIG not set, USER_CONFIG_PATH does NOT exist (Fallback to Default)
    with patch.dict(os.environ, {}, clear=True):
        with patch('os.path.exists', return_value=False):
            config = get_user_config()
            assert config == DEFAULT_CONFIG

    # Test 7: Edge case - config_file is exactly USER_CONFIG_PATH
    # The function logic says: if config_file and config_file is not USER_CONFIG_PATH
    # If they are the same, it falls through to the env var / existence check logic.
    with patch.dict(os.environ, {}, clear=True):
        with patch('os.path.exists', return_value=False):
            # This triggers the "User config not found" branch
            config = get_user_config(config_file=USER_CONFIG_PATH)
            assert config == DEFAULT_CONFIG
```


# LLM-generated content at query #13
#--------------------------

```python
import pytest
import yaml
import os
from pathlib import Path
from unittest.mock import patch, mock_open
from cookiecutter.exceptions import ConfigDoesNotExistException, InvalidConfiguration

def test_get_config(tmp_path):
    # Test Case 1: Config file does not exist
    non_existent_path = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException) as excinfo:
        get_config(non_existent_path)
    assert "does not exist" in str(excinfo.value)

    # Test Case 2: Valid config file with overrides
    config_data = {
        'cookiecutters_dir': '/tmp/custom_cookies',
        'abbreviations': {
            'gh': 'https://custom.com/{0}.git',
            'new_abbr': 'https://new.com/{0}'
        }
    }
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump(config_data), encoding='utf-8')

    config = get_config(config_file)

    # Verify top-level override
    assert config['cookiecutters_dir'] == '/tmp/custom_cookies'
    # Verify nested merge (abbreviations should have both old and new)
    assert config['abbreviations']['gh'] == 'https://custom.com/{0}.git'
    assert config['abbreviations']['new_abbr'] == 'https://new.com/{0}'
    # Verify preservation of defaults not in override
    assert 'gl' in config['abbreviations']
    assert config['replay_dir'] == os.path.expanduser('~/.cookiecutter_replay/')

    # Test Case 3: Invalid YAML syntax
    invalid_yaml_file = tmp_path / "invalid.yaml"
    invalid_yaml_file.write_text("key: : value", encoding='utf-8')
    with pytest.raises(InvalidConfiguration) as excinfo:
        get_config(invalid_yaml_file)
    assert "Unable to parse YAML file" in str(excinfo.value)

    # Test Case 4: YAML is not a dictionary (top-level list)
    list_yaml_file = tmp_path / "list.yaml"
    list_yaml_file.write_text("- item1\n- item2", encoding='utf-8')
    with pytest.raises(InvalidConfiguration) as excinfo:
        get_config(list_yaml_file)
    assert "should be an object" in str(excinfo.value)

    # Test Case 5: Empty YAML file (should return defaults)
    empty_yaml_file = tmp_path / "empty.yaml"
    empty_yaml_file.write_text("", encoding='utf-8')
    config_empty = get_config(empty_yaml_file)
    assert config_empty['cookiecutters_dir'] == os.path.expanduser('~/.cookiecutters/')
    assert config_empty['abbreviations']['bb'] == 'https://bitbucket.org/{0}'

    # Test Case 6: Path expansion via environment variables
    env_path_config = tmp_path / "env_path.yaml"
    env_path_config.write_text({'replay_dir': '$HOME/replay'}, encoding='utf-8')
    config_env = get_config(env_path_config)
    assert config_env['replay_dir'] == osар.path.expanduser('~/replay')
```


# LLM-generated content at query #14
#--------------------------

```python
import pytest
import yaml
from pathlib import Path
from unittest.mock import patch, mock_open
from cookiecutter.exceptions import ConfigDoesNotExistException, InvalidConfiguration

def test_get_config(tmp_path):
    """Test get_config with various scenarios: success, missing file, invalid YAML, and non-dict YAML."""
    
    # 1. Test Success Scenario
    config_data = {
        'cookiecutters_dir': '~/custom_cookies',
        'abbreviations': {
            'custom': 'https://custom.com/{0}'
        }
    }
    config_file = tmp_path / "config.yaml"
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config_data, f)

    config = get_config(config_file)
    
    # Verify merged values
    assert config['cookiecutters_dir'] == os.path.expanduser('~/custom_cookies')
    # Verify deep merge in abbreviations
    assert config['abbreviations']['gh'] == 'https://github.com/{0}.git'
    assert config['abbreviations']['custom'] == 'https://custom.com/{0}'
    # Verify default values preserved
    assert 'replay_dir' in config

    # 2. Test Config File Does Not Exist
    non_existent_path = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException) as excinfo:
        get_config(non_existent_path)
    assert "does not exist" in str(excinfo.value)

    # 3. Test Invalid YAML Syntax
    invalid_yaml_file = tmp_path / "invalid.yaml"
    with open(invalid_yaml_file, 'mock_open().write(') as f:
        # We use a real file with bad content for actual parsing error
        with open(invalid_yaml_file, 'w') as f:
            f.write("key: [unclosed bracket")
    
    with pytest.raises(InvalidConfiguration) as excinfo:
        get_config(invalid_yaml_file)
    assert "Unable to parse YAML file" in str(excinfo.value)

    # 4. Test YAML is not a dictionary (Top-level element check)
    list_yaml_file = tmp_path / "list.yaml"
    with open(list_yaml_file, 'w') as f:
        yaml.dump(['item1', 'item2'], f)
    
    with pytest.raises(InvalidConfiguration) as excinfo:
        get_config(list_yaml_file)
    assert "should be an object" in str(excinfo.value)

    # 5. Test Empty YAML file (should return defaults)
    empty_yaml_file = tmp_path / "empty.yaml"
    with open(empty_yaml_file, 'w') as f:
        f.write("")
    
    config_empty = get_config(empty_yaml_file)
    assert config_empty['cookiecutters_dir'] == os.path.expanduser('~/.cookiecutters/')
    assert config_empty['abbreviations']['gh'] == 'https://github.com/{0}.git'
```


# LLM-generated content at query #15
#--------------------------

```python
import pytest
import yaml
import os
from pathlib import Path
from unittest.mock import patch, mock_open

def test_get_config(tmp_path):
    # Test 1: Config file does not exist
    non_existent_path = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException) as excinfo:
        get_config(non_expected_path := str(non_existent_path))
    assert "does not exist" in str(excinfo.value)

    # Test 2: Valid config file with custom values
    config_data = {
        'cookiecutters_dir': '~/custom_cookiecutters/',
        'abbreviations': {
            'custom': 'https://custom.com/{0}'
        }
    }
    config_file = tmp_path / "valid_config.yaml"
    config_file.write_text(yaml.dump(config_data))

    config = get_config(str(config_file))

    # Check if custom values are applied
    assert config['cookiecutters_dir'] == os.path.expanduser('~/custom_cookiecutters/')
    # Check if abbreviations were merged (preserving BUILTIN_ABBREVIATIONS)
    assert config['abbreviations']['custom'] == 'https://custom.com/{0}'
    assert config['abbreviations']['gh'] == 'https://github.com/{0}.git'
    # Check if other defaults are preserved
    assert 'replay_dir' in config

    # Test 3: Invalid YAML syntax
    invalid_yaml_file = tmp_path / "invalid.yaml"
    invalid_yaml_file.write_text("key: : value: [unclosed bracket")
    with pytest.raises(InvalidConfiguration) as excinfo:
        get_config(str(invalid_yaml_file))
    assert "Unable to parse YAML file" in str(excinfo.value)

    # Test 4: YAML is not a dictionary (e.g., a list)
    list_yaml_file = tmp_path / "list.yaml"
    list_yaml_file.write_text("- item1\n- item2")
    with pytest.raises(InvalidConfiguration) as excinfo:
        get_config(str(list_yaml_file))
    assert "should be an object" in str(excinfo.value)

    # Test 5: Empty YAML file (should fallback to DEFAULT_CONFIG)
    empty_yaml_file = tmp_path / "empty.yaml"
    empty_yaml_file.write_text("")
    config_empty = get_config(str(empty_yaml_file))
    assert config_empty['cookiecutters_dir'] == DEFAULT_CONFIG['cookiecutters_dir']
    assert config_empty['abbreviations'] == BUILTIN_ABBREVIATIONS

    # Test 6: Testing path expansion via mock to avoid dependency on actual OS environment
    with patch('os.path.expandvars', return_value='$VAR_PATH'):
        with patch('os.path.expanduser', return_value='/expanded/path'):
            config_path_exp = tmp_path / "expand.yaml"
            config_path_exp.write_text("cookiecutters_dir: '$VAR_PATH'")
            config_exp = get_config(str(config_path_exp))
            assert config_exp['cookiecutters_dir'] == '/expanded/path'
```


####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
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

    # Test 2: Valid YAML file returns merged config
    config_file = tmp_path / "config.yaml"
    custom_content = {
        'cookiecutters_dir': '~/custom_cookies/',
        'abbreviations': {
            'custom': 'https://custom.com/{0}'
        }
    }
    config_file.write_text(yaml.dump(custom_content))
    
    config = get_config(config_file)
    
    # Check that custom values are applied
    assert config['cookiecutters_dir'] == os.path.expanduser('~/custom_cookies/')
    # Check that nested dicts (abbreviations) are merged, not overwritten
    assert config['abbreviations']['gh'] == 'https://github.com/{0}.git'
    assert config['abbreviations']['custom'] == 'https://custom.com/{0}'
    # Check that default values still exist
    assert 'replay_dir' in config

    # Test 3: Invalid YAML syntax
    invalid_yaml_file = tmp_path / "invalid.yaml"
    invalid_yaml_file.write_text("key: : value :")  # Invalid YAML
    with pytest.raises(InvalidConfiguration) as excinfo:
        get_config(invalid_yaml_file)
    assert "Unable to parse YAML file" in str(excinfo.value)

    # Test 4: YAML is not a dictionary (top-level list)
    list_yaml_file = tmp_path / "list.yaml"
    list_yaml_file.write_text("- item1\n- item2")
    with pytest.raises(InvalidConfiguration) as excinfo:
        get_config(list_yaml_file)
    assert "should be an object" in str(excinfo.value)

    # Test 5: Empty YAML file (should return defaults)
    empty_yaml_file = tmp_path / "empty.yaml"
    empty_yaml_file.write_text("")
    config = get_config(empty_yaml_file)
    assert config['cookiecutters_dir'] == os.path.expanduser('~/.cookiecutters/')

    # Test 6: Mocking open for precise control over content
    mock_yaml_content = "replay_dir: $ENV_VAR_PATH"
    # We use patch to simulate environment variable expansion during _expand_path
    with patch("builtins.open", mock_open(read_data=mock_yaml_content)):
        with patch("os.path.exists", return_value=True):
            with patch.dict(os.environ, {"ENV_VAR_PATH": "/tmp/env_path"}):
                config = get_config("mock_path.yaml")
                assert config['replay_dir'] == "/tmp/env_path"
```


# LLM-generated content at query #2
#--------------------------

```python
import os
import pytest
import yaml
from unittest.mock import patch, MagicMock

def test_get_user_config():
    # Setup common test data
    custom_config_content = {
        'cookiecutters_dir': '~/custom_templates',
        'abbreviations': {'gh': 'https://custom.com/{0}.git'}
    }
    
    # Test Case 1: default_config=True returns a copy of DEFAULT_CONFIG
    config = get_user_template_config_logic(default_config=True)
    assert config == DEFAULT_CONFIG
    assert config is not DEFAULT_CONFIG

    # Test Case 2: default_config is a dict, merges with DEFAULT_CONFIG
    overrides = {'replay_dir': '/tmp/replay', 'new_key': 'value'}
    config = get_user_template_config_logic(default_config=overrides)
    assert config['replay_dir'] == '/tmp/replay'
    assert config['new_key'] == 'value'
    assert config['abbreviations'] == BUILTIN_ABBREVIATIONS

    # Test Case 3: config_file is provided and valid
    with patch('os.path.exists', return_value=True), \
         patch('builtins.open', pytest.raises(Exception) if False else MagicMock()), \
         patch('yaml.safe_load', return_value=custom_config_content):
        
        # We mock open to return a file-like object containing our yaml
        with patch('builtins.open', unittest.mock.mock_open(read_data=yaml.dump(custom_config_content))):
            config = get_user_config(config_file='/fake/path.yaml')
            assert config['cookiecutters_dir'].endswith('custom_templates')
            assert config['abbreviations']['gh'] == 'https://custom.com/{0}.git'
            assert config['abbreviations']['gl'] == 'https://gitlab.com/{0}.git'

    # Test Case 4: COOKIECUTTER_CONFIG env var is set
    with patch.dict(os.environ, {'COOKIECUTTER_CONFIG': '/env/path.yaml'}), \
         patch('os.path.exists', return_value=True), \
         patch('builtins.open', unittest.mock.mock_open(read_data=yaml.dump({'replay_dir': '/env/replay'}))):
        config = get_user_config()
        assert config['replay_dir'].endswith('replay')

    # Test Case 5: No env var, no file path, no USER_CONFIG_PATH exists -> returns DEFAULT_CONFIG
    with patch.dict(os.environ, {}, clear=True), \
         patch('os.path.exists', return_value=False):
        config = get_user_config()
        assert config == DEFAULT_CONFIG

    # Test Case 6: No env var, USER_CONFIG_PATH exists -> loads USER_CONFIG_PATH
    with patch.dict(os.environ, {}, clear=True), \
         patch('os.path.exists', side_effect=lambda p: p == USER_CONFIG_PATH), \
         patch('builtins.open', unittest.mock.mock_open(read_data=yaml.dump({'replay_dir': '/user/replay'}))):
        config = get_user_config()
        assert config['replay_dir'].endswith('user/replay')

# Helper to allow the test to run without side effects on the actual environment
def get_user_template_config_logic(default_config=False, overrides=None):
    # This mirrors the logic of get_user_config but avoids env/file system dependency 
    # for the specific logic branches of the provided function signature.
    if default_config and isinstance(default_config, dict):
        return merge_configs(DEFAULT_CONFIG, default_config)
    if default_config:
        return copy.copy(DEFAULT_CONFIG)
    return None # Placeholder for the actual function call logic
```

Wait, the prompt asks for a single function `test_get_user_config`. Let me provide a clean, self-contained version.

```python
import os
import pytest
import yaml
from unittest.mock import patch, mock_open

def test_get_user_config():
    # 1. Test default_config=True (Returns default values)
    config = get_user_config(default_config=True)
    assert config == DEFAULT_CONFIG
    assert config is not DEFAULT_CONFIG

    # 2. Test default_config as a dict (Merges provided dict with defaults)
    overrides = {'replay_dir': '/tmp/replay', 'custom_key': 'custom_val'}
    config = get_user_config(default_config=overrides)
    assert config['replay_dir'] == os.path.expanduser('/tmp/replay')
    assert config['custom_key'] == 'custom_val'
    assert config['abbreviations'] == BUILTIN_ABBREVIATIONS

    # 3. Test providing a specific config_file path
    custom_yaml = yaml.dump({'cookiecutters_dir': '~/custom_dir', 'abbreviations': {'new': 'val'}})
    with patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data=custom_yaml)):
        config = get_user_config(config_file='/fake/path.yaml')
        assert config['cookiecutters_dir'].endswith('custom_dir')
        assert config['abbreviations']['new'] == 'val'
        assert config['abbreviations']['gh'] == BUILTIN_ABBREVIATIONS['gh']

    # 4. Test COOKIECUTTER_CONFIG environment variable
    custom_yaml_env = yaml.dump({'replay_dir': '/env/replay'})
    with patch.dict(os.environ, {'COOKIECUTTER_CONFIG': '/env/path.yaml'}), \
         patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data=custom_yaml_env)):
        config = get_user_config()
        assert config['replay_dir'].endswith('env/replay')

    # 5. Test fallback to USER_CONFIG_PATH when no env var and no file provided
    with patch.dict(os.environ, {}, clear=True), \
         patch('os.path.exists', side_effect=lambda p: p == USER_CONFIG_PATH), \
         patch('builtins.open', mock_open(read_data=yaml.dump({'replay_dir': '/user/replay'}))):
        config = get_user_config()
        assert config['replay_dir'].endswith('user/replay')

    # 6. Test fallback to DEFAULT_CONFIG when no config is found anywhere
    with patch.dict(os.environ, {}, clear=True), \
         patch('os.path.exists', return_value=False):
        config = get_user_config()
        assert config == DEFAULT_CONFIG

    # 7. Test InvalidConfiguration error for non-dict YAML
    with patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data="- item1\n- item2")):
        with pytest.raises(InvalidConfiguration):
            get_user_config(config_file='/fake/list.yaml')
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
    # Test 1: Config file does not exist
    non_existent_path = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException) as excinfo:
        get_config(non_existent_path)
    assert "does not exist" in str(excinfo.value)

    # Test 2: Config file contains invalid YAML
    invalid_yaml_path = tmp_path / "invalid.yaml"
    invalid_yaml_path.write_text("key: [unclosed bracket", encoding="utf-8")
    with pytest.raises(InvalidConfiguration) as excinfo:
        get_config(invalid_yaml_path)
    assert "Unable to parse YAML file" in str(excinfo.value)

    # Test 3: Config file contains top-level non-dict element (e.g., a list)
    list_yaml_path = tmp_path / "list.yaml"
    list_yaml_path.write_text("- item1\n- item2", encoding="utf-8")
    with pytest.raises(InvalidConfiguration) as excinfo:
        get_config(list_yaml_path)
    assert "should be an object" in str(excinfo.value)

    # Test 4: Successful config loading and merging with defaults
    valid_yaml_path = tmp_path / "valid.yaml"
    custom_content = {
        'replay_dir': '/tmp/custom_replay',
        'abbreviations': {
            'new_abbr': 'https://example.com/{0}'
        },
        'new_key': 'new_value'
    }
    valid_yaml_path.write_text(yaml.dump(custom_content), encoding="utf-8")

    config = get_config(valid_yaml_path)

    # Check if defaults are preserved
    assert config['cookiecutters_dir'] == os.path.expanduser('~/.cookiecutters/')
    
    # Check if overrides worked
    assert config['replay_dir'] == '/tmp/custom_replay'
    assert config['new_key'] == 'new_value'

    # Check if nested dictionary (abbreviations) was merged correctly
    # It should contain both the builtin 'gh' and the new 'new_abbr'
    assert 'gh' in config['abbreviations']
    assert config['abbreviations']['new_abbr'] == 'https://example.com/{0}'
    assert config['abbreviations']['gl'] == 'https://gitlab.com/{0}.git'

    # Test 5: Path expansion in config
    env_path_yaml = tmp_path / "env_path.yaml"
    env_path_yaml.write_text("cookiecutters_dir: $HOME/custom_dir", encoding="utf-8")
    config_expanded = get_config(env_path_yaml)
    expected_path = os.path.expanduser("$HOME/custom_dir")
    assert config_expanded['cookiecutters_dir'] == expected_path

    # Test 6: Mocking open for a clean test of the file reading logic
    mock_data = "cookiecutters_dir: /tmp/mock\n"
    with patch("builtins.open", mock_open(read_data=mock_data)):
        with patch("os.path.exists", return_value=True):
            config_mocked = get_config("dummy_path.yaml")
            assert config_mocked['cookiecutters_dir'] == "/tmp/mock"
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest
import os
import yaml
from unittest.mock import patch, MagicMock

def test_get_user_config():
    # Test 1: default_config as True (Return default values)
    config = get_user_config(default_config=True)
    assert config == DEFAULT_CONFIG
    assert config is not DEFAULT_CONFIG  # Should be a copy

    # Test 2: default_config as a dict (Merge provided values with defaults)
    custom_values = {'replay_dir': '/tmp/replay', 'new_key': 'new_val'}
    config = get_autofill_config_logic_helper(custom_values) # Using logic from get_user_config
    # Since we can't easily mock the internal logic without calling the function:
    config = get_user_config(default_config=custom_values)
    assert config['replay_dir'] == '/tmp/replay'
    assert config['new_key'] == 'new_val'
    assert config['abbreviations'] == BUILTIN_ABBREVIATIONS

    # Test 3: config_file provided (Load from custom path)
    custom_path = "/tmp/custom_config.yaml"
    custom_content = {'cookiecutters_dir': '/tmp/custom_cookies'}
    
    with patch('os.path.exists', return_value=True), \
         patch('builtins.open', MagicMock()), \
         patch('yaml.safe_load', return_value=custom_content), \
         patch('os.path.expandvars', side_effect=lambda x: x), \
         patch('os.path.expanduser', side_effect=lambda x: x):
        
        config = get_user_config(config_file=custom_path)
        assert config['cookiecutters_dir'] == '/tmp/custom_cookies'
        assert config['abbreviations'] == BUILTIN_ABBREVIATIONS

    # Test 4: COOKIECUTTER_CONFIG environment variable
    with patch.dict(os.environ, {'COOKIECUTTER_CONFIG': '/tmp/env_config.yaml'}), \
         patch('os.path.exists', return_value=True), \
         patch('builtins.open', MagicMock()), \
         patch('yaml.safe_load', return_value={'replay_dir': '/env/replay'}), \
         patch('os.path.expandvars', side_effect=lambda x: x), \
         patch('os.path.expanduser', side_effect=lambda x: x):
        
        config = get_user_config()
        assert config['replay_dir'] == '/env/replay'

    # Test 5: No config file, no env var, no user config file exists (Return defaults)
    with patch.dict(os.environ, {}, clear=True), \
         patch('os.path.exists', return_value=False):
        
        config = get_user_config()
        assert config == DEFAULT_CONFIG

    # Test 6: User config exists at USER_CONFIG_PATH (Default behavior)
    with patch.dict(os.environ, {}, clear=True), \
         patch('os.path.exists', side_effect=lambda p: p == USER_CONFIG_PATH), \
         patch('builtins.open', MagicMock()), \
         patch('yaml.safe_load', return_value={'abbreviations': {'new': 'path'}}), \
         patch('os.path.expandvars', side_effect=lambda x: x), \
         patch('os.path.expanduser', side_effect=lambda x: x):
        
        config = get_user_config()
        assert config['abbreviations']['new'] == 'path'
        assert config['abbreviations']['gh'] == 'https://github.com/{0}.git'

def get_autofill_config_logic_helper(val):
    # Helper to avoid duplication in test logic if needed, 
    # but here we just use the function directly.
    return get_user_config(default_config=val)
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest
import yaml
import os
from pathlib import Path

def test_get_config(tmp_path):
    # Test Case 1: Config file does not exist
    non_existent_path = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException) as excinfo:
        get_config(non_existent_path)
    assert "does not exist" in str(excinfo.value)

    # Test Case 2: Valid config file with simple overrides
    config_data = {
        'cookiecutters_dir': '/tmp/custom_cookies',
        'replay_dir': '/tmp/custom_replay'
    }
    config_file = tmp_path / "valid_config.yaml"
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config_data, f)
    
    config = get_config(config_file)
    assert config['cookiecutterns_dir'] == os.path.expanduser('/tmp/custom_cookies')
    assert config['replay_dir'] == os.path.expanduser('/tmp/custom_replay')
    # Ensure default values are preserved
    assert 'abbreviations' in config
    assert config['abbreviations']['gh'] == 'https://github.com/{0}.git'

    # Test Case 3: Config file with nested dictionary updates (abbreviations)
    nested_config_data = {
        'abbreviations': {
            'new_service': 'https://new.com/{0}'
        }
    }
    nested_config_file = tmp_path / "nested_config.yaml"
    with open(nested_config_file, 'w', encoding='utf-8') as f:
        yaml.dump(nested_config_data, f)
    
    config = get_config(nested_config_file)
    # Check that existing abbreviations are preserved and new one is added
    assert config['abbreviations']['gh'] == 'https://github.com/{0}.git'
    assert config['abbreviations']['new_service'] == 'https://new.com/{0}'

    # Test Case 4: Invalid YAML syntax
    invalid_yaml_file = tmp_path / "invalid.yaml"
    with open(invalid_yaml_file, 'w', encoding='utf-8') as f:
        f.write("key: [unclosed_bracket")
    
    with pytest.raises(InvalidConfiguration) as excinfo:
        get_config(invalid_yaml_file)
    assert "Unable to parse YAML file" in str(excinfo.value)

    # Test Case 5: YAML is valid but top-level is not a dictionary (e.g., a list)
    list_yaml_file = tmp_path / "list.yaml"
    with open(list_yaml_file, 'w', encoding='utf-8') as f:
        f.write("- item1\n- item2")
    
    with pytest.raises(InvalidConfiguration) as excinfo:
        get_config(list_yaml_file)
    assert "should be an object" in str(excinfo.value)

    # Test Case 6: Empty YAML file (should return default config)
    empty_yaml_file = tmp_path / "empty.yaml"
    with open(empty_yaml_file, 'w', encoding='utf-8') as f:
        f.write("")
    
    config = get_config(empty_yaml_file)
    assert config['cookiecutters_dir'] == os.path.expanduser(DEFAULT_CONFIG['cookiecutters_dir'])
    assert config['abbreviations'] == BUILTIN_ABBREVIATIONS

    # Test Case 7: Path expansion with environment variables
    env_var_path = f"$HOME/custom_env_path"
    env_config_file = tmp_path / "env_var.yaml"
    with open(env_config_file, 'w', encoding='utf-8') as f:
        yaml.dump({'cookiecutters_dir': env_var_path}, f)
    
    config = get_config(env_config_file)
    expected_path = os.path.expanduser(os.path.expandvars(env_var_path))
    assert config['cookiecutters_dir'] == expected_path
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest
import yaml
from pathlib import Path
from unittest.mock import patch, mock_open
from cookiecutter.exceptions import ConfigDoesNotExistException, InvalidConfiguration

def test_get_config(tmp_path):
    # Test Case 1: Config file does not exist
    non_existent_path = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException) as excinfo:
        get_config(non_existent_path)
    assert "does not exist" in str(excinfo.value)

    # Test Case 2: Valid config file with simple overrides
    config_data = {
        'cookiecutments_dir': '/tmp/custom_dir',
        'abbreviations': {
            'custom': 'https://custom.com/{0}'
        }
    }
    config_file = tmp_path / "valid_config.yaml"
    config_file.write_text(yaml.dump(config_data))

    config = get_config(config_file)
    
    # Check if top level was overwritten
    # Note: _expand_path is called, so /tmp/custom_dir remains /tmp/custom_dir (if not using ~)
    assert config['cookiecutters_dir'] == os.path.expanduser('/tmp/custom_dir')
    # Check if nested dict was merged (abbreviations should have gh, gl, bb AND custom)
    assert 'gh' in config['abbreviations']
    assert config['abbreviations']['custom'] == 'https://custom.com/{0}'

    # Test Case 3: Invalid YAML syntax
    invalid_yaml_file = tmp_path / "invalid.yaml"
    invalid_yaml_file.write_text("key: [unclosed_bracket")
    with pytest.raises(InvalidConfiguration) as excinfo:
        get_config(invalid_yaml_file)
    assert "Unable to parse YAML file" in str(excinfo.value)

    # Test Case 4: YAML is valid but not a dictionary (e.g., a list)
    list_yaml_file = tmp_path / "list.yaml"
    list_yaml_file.write_text("- item1\n- item2")
    with pytest.raises(InvalidConfiguration) as excinfo:
        get_config(list_yaml_file)
    assert "should be an object" in str(excinfo.value)

    # Test Case 5: Empty YAML file (should return defaults)
    empty_yaml_file = tmp_path / "empty.yaml"
    empty_yaml_file.write_text("")
    config_empty = get_config(empty_yaml_file)
    assert config_empty['abbreviations']['gh'] == BUILTIN_ABBREVIATIONS['gh']
    assert config_empty['replay_dir'] == os.path.expanduser('~/.cookiecutter_replay/')

    # Test Case 6: Path expansion check
    env_var_path = tmp_path / "env_path.yaml"
    env_var_path.write_text("cookiecutters_dir: '$HOME/test_dir'")
    config_env = get_config(env_var_path)
    assert config_env['cookiecutters_dir'] == os.path.expanduser('$HOME/test_dir')
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest
import os
from unittest.mock import patch, mock_open

def test_get_user_config():
    # Test Case 1: default_config is True (Return default values)
    config = get_user_config(default_config=True)
    assert config == DEFAULT_CONFIG
    assert config is not DEFAULT_CONFIG  # Should be a copy

    # Test Case 2: default_config is a dict (Merge provided dict with defaults)
    custom_values = {'replay_dir': '/tmp/replay', 'new_key': 'value'}
    config = get_user_annotated_config = get_user_config(default_config=custom_values)
    assert config['replay_dir'] == os.path.expanduser('/tmp/replay')
    assert config['new_key'] == 'value'
    assert config['abbreviations'] == BUILTIN_ABBREVIATIONS

    # Test Case 3: config_file is provided (Custom path)
    custom_path = "/tmp/custom_config.yaml"
    yaml_content = "cookiecutters_dir: '/tmp/custom_dir'\nabbreviations:\n  new: 'path'"
    
    with patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data=yaml_content)):
        config = get_user_config(config_file=custom_path)
        assert config['cookiecutters_dir'] == os.path.expanduser('/tmp/custom_dir')
        assert config['abbreviations']['new'] == 'path'
        assert config['abbreviations']['gh'] == BUILTIN_ABBREVIATIONS['gh']

    # Test Case 4: COOKIECUTTER_CONFIG env var is set
    env_path = "/tmp/env_config.yaml"
    env_yaml = "replay_dir: '$HOME/env_replay'"
    
    with patch.dict(os.environ, {'COOKIECUTTER_CONFIG': env_path}), \
         patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data=env_yaml)):
        config = get_user_config()
        assert config['replay_dir'] == os.path.expanduser('$HOME/env_replay')

    # Test Case 5: Fallback to USER_CONFIG_PATH (No env var, file exists)
    user_config_yaml = "cookiecutters_dir: '/tmp/user_dir'"
    with patch.dict(os.environ, {}, clear=True), \
         patch('os.path.exists', side_effect=lambda p: p == USER_CONFIG_PATH), \
         patch('builtins.open', mock_open(read_data=user_config_yaml)):
        config = get_user_config()
        assert config['cookiecutters_dir'] == os.path.expanduser('/tmp/user_dir')

    # Test Case 6: Fallback to DEFAULT_CONFIG (No env var, no user file)
    with patch.dict(os.environ, {}, clear=True), \
         patch('os.path.exists', return_value=False):
        config = get_user_config()
        assert config == DEFAULT_CONFIG

    # Test Case 7: Error handling - Invalid YAML in custom config
    with patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data=": invalid yaml")):
        with pytest.raises(InvalidConfiguration):
            get_user_config(config_file="/tmp/bad.yaml")

    # Test Case 8: Error handling - Config file does not exist
    with patch('os.path.exists', return_value=False):
        with pytest.raises(ConfigDoesNotExistException):
            get_user_config(config_file="/tmp/non_existent.yaml")
```


# LLM-generated content at query #8
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
    invalid_yaml_path.write_text("key: [unclosed bracket", encoding="utf-8")
    with pytest.raises(InvalidConfiguration) as excinfo:
        get_config(invalid_yaml_path)
    assert "Unable to parse YAML file" in str(excinfo.value)

    # Test 3: YAML is not a dictionary (top-level is a list)
    list_yaml_path = tmp_path / "list.yaml"
    list_yaml_path.write_text("- item1\n- item2", encoding="utf-8")
    with pytest.raises(InvalidConfiguration) as excinfo:
        get_config(list_yaml_path)
    assert "should be an object" in str(excinfo.value)

    # Test 4: Successful load and merge
    valid_yaml_path = tmp_path / "valid.yaml"
    custom_content = {
        'cookiecutters_dir': '/custom/path/cookies/',
        'abbreviations': {
            'new_key': 'https://new.com/{0}'
        }
    }
    valid_yaml_path.write_text(yaml.dump(custom_content), encoding="utf-8")

    config = get_config(valid_yaml_path)

    # Check that custom values are applied
    assert config['cookiecutters_dir'] == os.path.expanduser('/custom/path/cookies/')
    # Check that abbreviations are merged (preserved old, added new)
    assert config['abbreviations']['gh'] == 'https://github.com/{0}.git'
    assert config['abbreviations']['new_key'] == 'https://new.com/{0}'
    # Check that default values are preserved
    assert 'replay_dir' in config
    assert config['replay_dir'] == os.path.expanduser('~/.cookiecutter_replay/')

    # Test 5: Mocking file open for edge case testing (empty file)
    with patch("builtins.open", mock_open(read_data="")):
        with patch("os.path.exists", return_value=True):
            config_empty = get_config(valid_yaml_path)
            # Should return DEFAULT_CONFIG because empty yaml loads as None/{}
            assert config_empty['abbreviations']['gh'] == 'https://github.com/{0}.git'
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest
import yaml
import os
from pathlib import Path
from unittest.mock import patch, mock_open
from cookiecutter.exceptions import ConfigDoesNotExistException, InvalidConfiguration

def test_get_config(tmp_path):
    # Test Case 1: Config file does not exist
    non_existent_path = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException) as excinfo:
        get_config(non_existent_path)
    assert "does not exist" in str(excinfo.value)

    # Test Case 2: Valid config file with overrides
    config_data = {
        'cookiecutters_dir': '/tmp/custom_dir/',
        'abbreviations': {
            'custom': 'https://custom.com/{0}'
        }
    }
    config_file = tmp_path / "valid_config.yaml"
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config_data, f)

    config = get_config(config_file)
    
    # Check if overrides worked
    assert config['cookiecutters_dir'] == os.path.expanduser('/tmp/custom_dir/')
    # Check if nested merge worked (abbreviations should contain both builtin and custom)
    assert 'gh' in config['abbreviations']
    assert config['abbreviations']['custom'] == 'https://custom.com/{0}'
    # Check if defaults are preserved
    assert 'replay_dir' in config

    # Test Case 3: Invalid YAML syntax
    invalid_yaml_file = tmp_path / "invalid.yaml"
    with open(invalid_yaml_file, 'w', encoding='utf-8') as f:
        f.write("invalid: [unclosed bracket")
    
    with pytest.raises(InvalidConfiguration) as excinfo:
        get_config(invalid_yaml_file)
    assert "Unable to parse YAML file" in str(excinfo.value)

    # Test Case 4: YAML is not a dictionary (top-level is a list)
    list_yaml_file = tmp_path / "list.yaml"
    with open(list_yaml_file, 'w', encoding='utf-8') as f:
        f.write("- item1\n- item2")
    
    with pytest.raises(InvalidConfiguration) as excinfo:
        get_config(list_yaml_file)
    assert "should be an object" in str(excinfo.value)

    # Test Case 5: Empty YAML file (should return defaults)
    empty_yaml_file = tmp_path / "empty.yaml"
    with open(empty_yaml_file, 'w', encoding='utf-8') as f:
        f.write("")
    
    config_empty = get_config(empty_yaml_file)
    assert config_empty['cookiecutters_dir'] == DEFAULT_CONFIG['cookiecutters_dir']
    assert config_empty['abbreviations'] == BUILTIN_ABBREVIATIONS

    # Test Case 6: Environment variable expansion in paths
    env_path_file = tmp_path / "env_path.yaml"
    with patch.dict(os.environ, {'MY_TEST_DIR': '/tmp/env_test'}):
        with open(env_path_file, 's', encoding='utf-8') as f:
            # Using a string that contains an env var
            yaml.dump({'cookiecutters_dir': '$MY_TEST_DIR/cookies'}, f)
        
        config_env = get_config(env_path_file)
        assert config_env['cookiecutters_dir'] == os.path.expanduser('/tmp/env_test/cookies')
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest
import yaml
import os
from pathlib import Path
from unittest.mock import patch, mock_open
from cookiecutter.exceptions import ConfigDoesNotExistException, InvalidConfiguration

def test_get_config(tmp_path):
    # Test Case 1: Config file does not exist
    non_existent_path = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException) as excinfo:
        get_config(non_existent_path)
    assert "does not exist" in str(excinfo.value)

    # Test Case 2: Valid config file with overrides
    config_data = {
        'cookiecutters_dir': '/tmp/custom_cookiecutters/',
        'abbreviations': {
            'custom': 'https://custom.com/{0}'
        }
    }
    config_file = tmp_path / "config.yaml"
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config_data, f)

    config = get_config(config_file)
    
    # Check that overrides worked
    assert config['cookiecutters_dir'] == os.path.expanduser('/tmp/custom_cookiecutters/')
    # Check that nested dict merge preserved BUILTIN_ABBREVIATIONS
    assert config['abbreviations']['gh'] == 'https://github.com/{0}.git'
    assert config['abbreviations']['custom'] == 'https://custom.com/{0}'
    # Check that default values were preserved
    assert 'replay_dir' in config

    # Test Case 3: Invalid YAML syntax
    invalid_yaml_file = tmp_path / "invalid.yaml"
    with open(invalid_yaml_file, 'w', encoding='utf-8') as f:
        f.write("key: : invalid")
    
    with pytest.raises(InvalidConfiguration) as excinfo:
        get_config(invalid_yaml_file)
    assert "Unable to parse YAML file" in str(excinfo.value)

    # Test Case 4: YAML is not a dictionary (top-level list)
    list_yaml_file = tmp_path / "list.yaml"
    with open(list_yaml_file, 'w', encoding='utf-8') as f:
        yaml.dump(['item1', 'item2'], f)
    
    with pytest.raises(InvalidConfiguration) as excinfo:
        get_config(list_yaml_file)
    assert "should be an object" in str(excinfo.value)

    # Test Case 5: Empty YAML file (should return default config)
    empty_yaml_file = tmp_path / "empty.yaml"
    with open(empty_yaml_file, 'w', encoding='utf-8') as f:
        f.write("")
    
    config_empty = get_config(empty_yaml_file)
    assert config_empty['cookiecutters_dir'] == os.path.expanduser(DEFAULT_CONFIG['cookiecutters_dir'])
    assert config_empty['abbreviations'] == BUILTIN_ABBREVIATIONS
```


# LLM-generated content at query #11
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
    invalid_yaml_path.write_text("key: [unclosed bracket", encoding="utf-8")
    with pytest.raises(InvalidConfiguration) as excinfo:
        get_config(invalid_yaml_path)
    assert "Unable to parse YAML file" in str(excinfo.value)

    # Test 3: YAML is not a dictionary (top-level element is list)
    list_yaml_path = tmp_path / "list.yaml"
    list_yaml_path.write_text("- item1\n- item2", encoding="utf-8")
    with pytest.raises(InvalidConfiguration) as excinfo:
        get_config(list_yaml_path)
    assert "should be an object" in str(excinfo.value)

    # Test 4: Successful config loading and merging
    valid_yaml_path = tmp_path / "valid.yaml"
    custom_content = {
        'cookiecutters_dir': '/tmp/custom_cookies/',
        'abbreviations': {
            'new_abbr': 'https://custom.com/{0}'
        }
    }
    with open(valid_yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(custom_content, f)

    config = get_config(valid_yaml_path)

    # Check that custom values are applied
    assert config['cookiecutters_dir'] == os.path.expanduser('/tmp/custom_cookies/')
    
    # Check that nested dicts (abbreviations) are merged, not overwritten
    assert config['abbreviations']['gh'] == 'https://github.com/{0}.git'
    assert config['abbreviations']['new_abbr'] == 'https://custom.com/{0}'
    
    # Check that default values are preserved
    assert 'replay_dir' in config
    assert config['replay_dir'] == os.path.expanduser('~/.cookiecutter_replay/')

    # Test 5: Testing path expansion via mock for environment variables
    with patch.dict(os.environ, {"MY_VAR": "expanded_val"}):
        env_path_yaml = tmp_path / "env_path.yaml"
        env_path_content = {'replay_dir': '$MY_VAR/replay'}
        with open(env_path_yaml, 'w', encoding='utf-8') as f:
            yaml.dump(env_path_content, f)
        
        config_env = get_config(env_path_yaml)
        assert 'expanded_val/replay' in config_env['replay_dir']

    # Test 6: Empty YAML file should return defaults
    empty_yaml_path = tmp_path / "empty.yaml"
    empty_yaml_path.write_text("", encoding="utf-8")
    config_empty = get_config(empty_yaml_path)
    assert config_empty['cookiecutters_dir'] == DEFAULT_CONFIG['cookiecutters_dir']
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest
import yaml
import os
from pathlib import Path
from unittest.mock import patch, mock_open

def test_get_config():
    # Test Case 1: Config file does not exist
    with pytest.raises(ConfigDoesNotExistException) as excinfo:
        get_config("/non/existent/path/config.yaml")
    assert "does not exist" in str(excinfo.value)

    # Test Case 2: YAML is invalid (syntax error)
    invalid_yaml_content = "key: [unclosed list"
    with patch("os.path.exists", return_value=True):
        with patch("builtins.open", mock_open(read_data=invalid_yaml_content)):
            with pytest.raises(InvalidConfiguration) as excinfo:
                get_config("config.yaml")
            assert "Unable to parse YAML file" in str(excinfo.value)

    # Test Case 3: YAML is valid but top-level is not a dictionary (e.g., a list)
    list_yaml_content = "- item1\n- item2"
    with patch("os.path.exists", return_value=True):
        with patch("builtins.open", mock_open(read_data=list_yaml_content)):
            with pytest.raises(InvalidConfiguration) as excinfo:
                get_config("config.yaml")
            assert "should be an object" in str(excinfo.value)

    # Test Case 4: Valid config with overrides and path expansion
    # We mock expandvars and expanduser to ensure predictable output for testing
    valid_yaml_content = """
cookiecutters_dir: "$HOME/custom_cookies"
abbreviations:
  gh: "https://custom-github.com/{0}.git"
replay_dir: "/tmp/replay"
"""
    # Mocking the environment and path expansion behavior
    # We use a patch on _expand_path to avoid dependency on the actual host OS environment
    with patch("os.path.exists", return_value=True):
        with patch("builtins.open", mock_open(read_data=valid_yaml_content)):
            with patch("_expand_path", side_effect=lambda x: x.replace("$HOME", "/home/user")):
                config = get_config("config.yaml")
                
                # Verify merged abbreviations (gh is overridden, bb/gl remain from DEFAULT)
                assert config['abbreviations']['gh'] == "https://custom-github.com/{0}.git"
                assert config['abbreviations']['bb'] == 'https://bitbucket.org/{0}'
                
                # Verify overridden paths
                assert config['cookiecutters_dir'] == "/home/user/custom_cookies"
                assert config['replay_dir'] == "/tmp/replay"
                
                # Verify default values are preserved
                assert 'default_context' in config
                assert config['default_context'] == collections.OrderedDict([])

    # Test Case 5: Empty YAML file (should return DEFAULT_CONFIG with expanded paths)
    with patch("os.path.exists", return_value=True):
        with patch("builtins.open", mock_open(read_data="")):
            with patch("_expand_path", side_effect=lambda x: x):
                config = get_config("empty.yaml")
                assert config['abbreviations']['gl'] == 'https://gitlab.com/{0}.git'
                assert config['cookiecutters_dir'] == DEFAULT_CONFIG['cookiecutters_dir']
```


# LLM-generated content at query #13
#--------------------------

```python
import pytest
import yaml
import os
from pathlib import Path
from unittest.mock import patch, mock_open
from cookiecutter.exceptions import ConfigDoesNotExistException, InvalidConfiguration

def test_get_config(tmp_path):
    # Test 1: Config file does not exist
    non_existent_path = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException) as excinfo:
        get_config(non_existent_path)
    assert "does not exist" in str(excinfo.value)

    # Test 2: Config file is valid YAML and merges correctly
    config_data = {
        'cookiecutters_dir': '~/custom_templates',
        'abbreviations': {
            'custom': 'https://custom.com/{0}'
        }
    }
    config_file = tmp_path / "valid_config.yaml"
    config_file.write_text(yaml.dump(config_data))
    
    config = get_config(config_file)
    
    # Check if merged correctly (abbreviations should contain both builtin and custom)
    assert 'gh' in config['abbreviations']
    assert config['abbreviations']['custom'] == 'https://custom.com/{0}'
    # Check if path expansion worked
    assert config['cookiecutters_dir'].endswith('custom_templates')
    # Check if default value remains for non-overridden keys
    assert 'replay_dir' in config

    # Test 3: Config file contains invalid YAML
    invalid_yaml_file = tmp_path / "invalid.yaml"
    invalid_yaml_file.write_text("key: : value") # Invalid syntax
    with pytest.raises(InvalidConfiguration) as excinfo:
        get_config(invalid_yaml_file)
    assert "Unable to parse YAML" in str(excinfo.value)

    # Test 4: Config file is valid YAML but not a dictionary (e.g., a list)
    list_yaml_file = tmp_path / "list.yaml"
    list_yaml_file.write_text("- item1\n- item2")
    with pytest.raises(InvalidConfiguration) as excinfo:
        get_config(list_yaml_file)
    assert "should be an object" in str(excinfo.value)

    # Test 5: Config file is empty
    empty_yaml_file = tmp_path / "empty.yaml"
    empty_yaml_file.write_text("")
    config_empty = get_config(empty_yaml_file)
    # Should return DEFAULT_CONFIG with expanded paths
    assert config_empty['abbreviations'] == BUILTIN_ABBREVIATIONS
    assert config_empty['replay_dir'] == os.path.expanduser('~/.cookiecutter_replay/')

    # Test 6: Mocking open for a controlled environment (no disk dependency)
    mock_yaml_content = "new_key: new_value"
    with patch("builtins.open", mock_open(read_data=mock_yaml_content)):
        with patch("os.path.exists", return_value=True):
            config_mock = get_config("dummy_path.yaml")
            assert config_mock['new_key'] == 'new_value'
            assert config_mock['abbreviations'] == BUILTIN_ABBREVIATIONS
```


# LLM-generated content at query #14
#--------------------------

```python
import pytest
import yaml
import os
from pathlib import Path
from unittest.mock import patch, mock_open
from cookiecutter.exceptions import ConfigDoesNotExistException, InvalidConfiguration

def test_get_config(tmp_path):
    # Test 1: Config file does not exist
    non_existent_path = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigDoesNotExistException) as excinfo:
        get_config(non_existent_path)
    assert "does not exist" in str(excinfo.value)

    # Test 2: Valid config file with simple values
    config_data = {
        'cookiecutters_dir': '/tmp/custom_dir/',
        'replay_dir': '/tmp/custom_replay/'
    }
    config_file = tmp_path / "valid_config.yaml"
    config_file.write_text(yaml.dump(config_data))
    
    config = get_config(config_file)
    assert config['cookietters_dir'] == os.path.expanduser('/tmp/custom_dir/')
    assert config['replay_dir'] == os.path.expanduser('/tmp/custom_replay/')
    # Ensure defaults are merged
    assert config['abbreviations'] == BUILTIN_ABBREVIATIONS

    # Test 3: Valid config file with nested dictionary (abbreviations)
    nested_config_data = {
        'abbreviations': {
            'new_abbr': 'https://newsite.com/{0}'
        }
    }
    nested_config_file = tmp_path / "nested_config.yaml"
    nested_config_file.write_text(yaml.dump(nested_config_data))
    
    config = get_config(nested_config_file)
    # Check that existing abbreviations are preserved and new one is added
    assert config['abbreviations']['gh'] == 'https://github.com/{0}.git'
    assert config['abbreviations']['new_abbr'] == 'https://newsite.com/{0}'

    # Test 4: Invalid YAML syntax
    invalid_yaml_file = tmp_path / "invalid.yaml"
    invalid_yaml_file.write_text("key: [unclosed bracket")
    with pytest.raises(InvalidConfiguration) as excinfo:
        get_config(invalid_yaml_file)
    assert "Unable to parse YAML file" in str(excinfo.value)

    # Test 5: YAML is not a dictionary (e.g., a list)
    list_yaml_file = tmp_path / "list.yaml"
    list_yaml_file.write_text("- item1\n- item2")
    with pytest.raises(InvalidConfiguration) as excinfo:
        get_config(list_yaml_file)
    assert "should be an object" in str(excinfo.value)

    # Test 6: Empty YAML file (should return default config)
    empty_yaml_file = tmp_path / "empty.yaml"
    empty_yaml_file.write_text("")
    config = get_config(empty_yaml_file)
    assert config['cookiecutters_dir'] == os.path.expanduser(DEFAULT_CONFIG['cookiecutters_dir'])

    # Test 7: Environment variable expansion in paths
    env_path_config = tmp_path / "env_path.yaml"
    env_path_config.write_text("cookiecutters_dir: $HOME/custom_env_dir")
    # Mocking os.environ to ensure $HOME is predictable for the test
    with patch.dict(os.environ, {'HOME': '/tmp/user'}):
        config = get_config(env_path_config)
        assert config['cookiecutters_dir'] == os.path.expanduser('/tmp/user/custom_env_dir')
```


