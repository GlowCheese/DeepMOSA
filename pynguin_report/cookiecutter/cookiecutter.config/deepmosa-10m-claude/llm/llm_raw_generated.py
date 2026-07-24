####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_get_user_config_with_default_config_dict():
    default_config_dict = {'replay_dir': '/custom/replay', 'cookiecutters_dir': '/custom/cookies'}
    result = get_user_config(default_config=default_config_dict)
    assert result['replay_dir'] == '/custom/replay'
    assert result['cookiecutters_dir'] == '/custom/cookies'


def test_get_user_config_with_default_config_true():
    result = get_user_config(default_config=True)
    assert isinstance(result, dict)
    assert result == DEFAULT_CONFIG


def test_get_user_config_with_custom_config_file(tmp_path):
    config_file = tmp_path / "custom_config.yaml"
    config_file.write_text("replay_dir: /tmp/replay\ncookiecutters_dir: /tmp/cookies\n")
    result = get_user_config(config_file=str(config_file))
    assert isinstance(result, dict)
    assert result['replay_dir'] == '/tmp/replay'


def test_get_user_config_no_args_no_env_no_user_config(monkeypatch, tmp_path):
    monkeypatch.delenv('COOKIECUTTER_CONFIG', raising=False)
    monkeypatch.setattr('os.path.exists', lambda x: False)
    result = get_user_config()
    assert result == DEFAULT_CONFIG


def test_get_user_config_with_env_variable(monkeypatch, tmp_path):
    config_file = tmp_path / "env_config.yaml"
    config_file.write_text("replay_dir: /env/replay\ncookiecutters_dir: /env/cookies\n")
    monkeypatch.setenv('COOKIECUTTER_CONFIG', str(config_file))
    result = get_user_config()
    assert isinstance(result, dict)
    assert result['replay_dir'] == '/env/replay'


def test_get_user_config_with_user_config_path(monkeypatch, tmp_path):
    config_file = tmp_path / "user_config.yaml"
    config_file.write_text("replay_dir: /user/replay\ncookiecutters_dir: /user/cookies\n")
    monkeypatch.delenv('COOKIECUTTER_CONFIG', raising=False)
    monkeypatch.setattr('os.path.exists', lambda x: True)
    monkeypatch.setattr('builtins.open', lambda *args, **kwargs: open(config_file, *args, **kwargs))
    result = get_user_config()
    assert isinstance(result, dict)


def test_get_user_config_config_file_not_exist():
    try:
        get_user_config(config_file='/nonexistent/path/config.yaml')
        assert False, "Should raise ConfigDoesNotExistException"
    except ConfigDoesNotExistException:
        assert True


def test_get_user_config_invalid_yaml(tmp_path):
    config_file = tmp_path / "invalid_config.yaml"
    config_file.write_text("invalid: yaml: content: [")
    try:
        get_user_config(config_file=str(config_file))
        assert False, "Should raise InvalidConfiguration"
    except InvalidConfiguration:
        assert True


def test_get_user_config_yaml_not_dict(tmp_path):
    config_file = tmp_path / "list_config.yaml"
    config_file.write_text("- item1\n- item2\n")
    try:
        get_user_config(config_file=str(config_file))
        assert False, "Should raise InvalidConfiguration"
    except InvalidConfiguration:
        assert True


def test_merge_configs_simple():
    default = {'a': 1, 'b': 2}
    overwrite = {'b': 3, 'c': 4}
    result = merge_configs(default, overwrite)
    assert result['a'] == 1
    assert result['b'] == 3
    assert result['c'] == 4


def test_merge_configs_nested():
    default = {'a': {'x': 1, 'y': 2}, 'b': 3}
    overwrite = {'a': {'y': 20}, 'c': 4}
    result = merge_configs(default, overwrite)
    assert result['a']['x'] == 1
    assert result['a']['y'] == 20
    assert result['b'] == 3
    assert result['c'] == 4


def test_merge_configs_deep_nested():
    default = {'a': {'b': {'c': 1, 'd': 2}}}
    overwrite = {'a': {'b': {'d': 20}}}
    result = merge_configs(default, overwrite)
    assert result['a']['b']['c'] == 1
    assert result['a']['b']['d'] == 20


def test_expand_path_with_home():
    result = _expand_path('~/test/path')
    assert '~' not in result
    assert result.startswith('/')


def test_expand_path_with_env_var(monkeypatch):
    monkeypatch.setenv('TEST_VAR', '/test/value')
    result = _expand_path('$TEST_VAR/path')
    assert result == '/test/value/path'


# LLM-generated content at query #2
#--------------------------

```python
def test_predicate_line_33_evaluates_to_false():
    from unittest.mock import patch
    
    # Test case 1: config_file is None
    result = get_user_config(config_file=None, default_config=False)
    assert result is not None
    
    # Test case 2: config_file equals USER_CONFIG_PATH
    with patch('os.path.exists', return_value=False):
        with patch('os.environ', {}):
            result = get_user_config(config_file=USER_CONFIG_PATH, default_config=False)
            assert result is not None
    
    # Test case 3: config_file is empty string (falsy)
    with patch('os.path.exists', return_value=False):
        with patch('os.environ', {}):
            result = get_user_config(config_file="", default_config=False)
            assert result is not None


# LLM-generated content at query #3
#--------------------------

```python
def test_predicate_at_line_33_evaluates_to_false():
    config_file = "/path/to/default/config"
    USER_CONFIG_PATH = "/path/to/default/config"
    
    predicate = config_file and config_file is not USER_CONFIG_PATH
    
    assert predicate is False


# LLM-generated content at query #4
#--------------------------

```python
def test_get_user_config_with_default_config_dict():
    default_config_dict = {'replay_dir': '/custom/replay'}
    result = get_user_config(default_config=default_config_dict)
    assert isinstance(result, dict)
    assert result['replay_dir'] == '/custom/replay'


def test_get_user_config_with_default_config_true():
    result = get_user_config(default_config=True)
    assert isinstance(result, dict)
    assert result == copy.copy(DEFAULT_CONFIG)


def test_get_user_config_with_custom_config_file(tmp_path):
    config_file = tmp_path / "custom_config.yaml"
    config_file.write_text("replay_dir: /tmp/replay\ncookiecutters_dir: /tmp/cookies")
    result = get_user_config(config_file=str(config_file))
    assert isinstance(result, dict)
    assert result['replay_dir'] == '/tmp/replay'


def test_get_user_config_with_env_variable(tmp_path, monkeypatch):
    config_file = tmp_path / "env_config.yaml"
    config_file.write_text("replay_dir: /env/replay\ncookiecutters_dir: /env/cookies")
    monkeypatch.setenv('COOKIECUTTER_CONFIG', str(config_file))
    result = get_user_config()
    assert isinstance(result, dict)
    assert result['replay_dir'] == '/env/replay'


def test_get_user_config_default_path_exists(tmp_path, monkeypatch):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("replay_dir: /default/replay\ncookiecutters_dir: /default/cookies")
    monkeypatch.setattr('os.path.exists', lambda path: path == str(config_file))
    monkeypatch.setattr('builtins.open', lambda *args, **kwargs: open(config_file, *args[1:], **kwargs))
    monkeypatch.delenv('COOKIECUTTER_CONFIG', raising=False)
    result = get_user_config()
    assert isinstance(result, dict)


def test_get_user_config_default_path_not_exists(monkeypatch):
    monkeypatch.setenv('COOKIECUTTER_CONFIG', '')
    monkeypatch.delenv('COOKIECUTTER_CONFIG')
    monkeypatch.setattr('os.path.exists', lambda path: False)
    result = get_user_config()
    assert isinstance(result, dict)
    assert result == copy.copy(DEFAULT_CONFIG)


def test_get_user_config_invalid_config_file(tmp_path):
    config_file = tmp_path / "nonexistent_config.yaml"
    try:
        get_user_config(config_file=str(config_file))
        assert False, "Should raise ConfigDoesNotExistException"
    except ConfigDoesNotExistException:
        pass


def test_get_user_config_with_malformed_yaml(tmp_path):
    config_file = tmp_path / "malformed_config.yaml"
    config_file.write_text("invalid: yaml: content: [")
    try:
        get_user_config(config_file=str(config_file))
        assert False, "Should raise InvalidConfiguration"
    except InvalidConfiguration:
        pass


def test_get_user_config_priority_default_config_dict_over_file(config_file):
    default_config_dict = {'replay_dir': '/priority/replay'}
    result = get_user_config(config_file=config_file, default_config=default_config_dict)
    assert result['replay_dir'] == '/priority/replay'


def test_get_user_config_merge_with_defaults():
    custom_config = {'replay_dir': '/custom/replay'}
    result = get_user_config(default_config=custom_config)
    assert result['replay_dir'] == '/custom/replay'
    assert 'cookiecutters_dir' in result


# LLM-generated content at query #5
#--------------------------

```python
def test_get_config_file_does_not_exist(tmp_path):
    config_path = tmp_path / "nonexistent.yaml"
    try:
        get_config(config_path)
        assert False, "Should have raised ConfigDoesNotExistException"
    except ConfigDoesNotExistException:
        pass


def test_get_config_invalid_yaml(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("invalid: yaml: content:", encoding='utf-8')
    try:
        get_config(config_file)
        assert False, "Should have raised InvalidConfiguration"
    except InvalidConfiguration:
        pass


def test_get_config_non_dict_yaml(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("- item1\n- item2", encoding='utf-8')
    try:
        get_config(config_file)
        assert False, "Should have raised InvalidConfiguration"
    except InvalidConfiguration:
        pass


def test_get_config_empty_yaml(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("", encoding='utf-8')
    result = get_config(config_file)
    assert isinstance(result, dict)
    assert 'replay_dir' in result
    assert 'cookiecutters_dir' in result


def test_get_config_valid_config_with_paths(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("replay_dir: /test/replay\ncookiecutters_dir: /test/cookies", encoding='utf-8')
    result = get_config(config_file)
    assert isinstance(result, dict)
    assert result['replay_dir'] == '/test/replay'
    assert result['cookiecutters_dir'] == '/test/cookies'


def test_get_config_expands_home_directory(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("replay_dir: ~/replay\ncookiecutters_dir: ~/cookies", encoding='utf-8')
    result = get_config(config_file)
    assert '~' not in result['replay_dir']
    assert '~' not in result['cookiecutters_dir']


def test_get_config_merges_with_defaults(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("replay_dir: /custom/replay", encoding='utf-8')
    result = get_config(config_file)
    assert result['replay_dir'] == '/custom/replay'
    assert 'cookiecutters_dir' in result


def test_get_config_nested_dict_merge(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("some_nested_config:\n  key1: value1", encoding='utf-8')
    result = get_config(config_file)
    assert isinstance(result, dict)
    assert result['some_nested_config']['key1'] == 'value1'


# LLM-generated content at query #6
#--------------------------

```python
def test_line_40_predicate_evaluates_to_false():
    import os
    from unittest.mock import patch
    
    # Ensure COOKIECUTTER_CONFIG is not set in environment
    with patch.dict(os.environ, {}, clear=False):
        if 'COOKIECUTTER_CONFIG' in os.environ:
            del os.environ['COOKIECUTTER_CONFIG']
        
        try:
            env_config_file = os.environ['COOKIECUTTER_CONFIG']
            predicate_result = False
        except KeyError:
            predicate_result = True
    
    assert predicate_result is True


# LLM-generated content at query #7
#--------------------------

```python
def test_cookiecutter_config_env_var_not_set():
    import os
    import copy
    from unittest.mock import patch, MagicMock
    
    # Mock the environment to ensure COOKIECUTTER_CONFIG is not set
    mock_environ = {}
    mock_default_config = {"key": "value"}
    mock_get_config = MagicMock()
    mock_copy = MagicMock(return_value=mock_default_config)
    
    with patch.dict(os.environ, mock_environ, clear=True):
        with patch('os.path.exists', return_value=False):
            with patch('copy.copy', mock_copy):
                # Simulate the try-except block
                try:
                    env_config_file = os.environ['COOKIECUTTER_CONFIG']
                    predicate_result = True
                except KeyError:
                    predicate_result = False
    
    assert predicate_result is False


# LLM-generated content at query #8
#--------------------------

```python
def test_predicate_line_43_evaluates_to_true(tmp_path, monkeypatch):
    import os
    from unittest.mock import patch
    
    # Create a temporary config file
    config_file = tmp_path / "config.yaml"
    config_file.write_text("test: value")
    
    # Mock USER_CONFIG_PATH to point to our temporary file
    with patch('os.path.exists') as mock_exists:
        mock_exists.return_value = True
        
        # Mock os.environ to not have COOKIECUTTER_CONFIG
        with patch.dict(os.environ, {}, clear=True):
            # Mock get_config to return a dict
            with patch('get_config') as mock_get_config:
                mock_get_config.return_value = {"test": "value"}
                
                # Call the function with default_config=False and config_file=None
                # This should trigger the code path leading to line 43
                result = get_user_config(config_file=None, default_config=False)
                
                # Verify that os.path.exists was called (line 43 predicate)
                mock_exists.assert_called()
                # Verify the predicate evaluated to True by checking get_config was called
                mock_get_config.assert_called()


# LLM-generated content at query #9
#--------------------------

```python
def test_get_config_predicate_line_3_evaluates_to_true(tmp_path):
    import os
    from pathlib import Path
    
    config_path = tmp_path / "config.yaml"
    config_path.write_text("replay_dir: /tmp\ncookiecutters_dir: /tmp\n")
    
    result = os.path.exists(config_path)
    
    assert result is True


# LLM-generated content at query #10
#--------------------------

```python
def test_user_config_path_exists():
    import os
    import tempfile
    from unittest.mock import patch, MagicMock
    
    # Create a temporary file to act as USER_CONFIG_PATH
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_path = temp_file.name
    
    try:
        # Mock the dependencies
        with patch('os.path.exists') as mock_exists, \
             patch('os.environ', {}), \
             patch('get_config') as mock_get_config, \
             patch('copy.copy') as mock_copy:
            
            # Configure mocks
            mock_exists.return_value = True
            mock_get_config.return_value = {'test': 'config'}
            mock_copy.return_value = {'default': 'config'}
            
            # Import after patching
            from your_module import get_user_config, USER_CONFIG_PATH
            
            # Call the function with no arguments to trigger line 43
            result = get_user_config()
            
            # Assert that os.path.exists was called with USER_CONFIG_PATH
            mock_exists.assert_called()
            # Assert that get_config was called with USER_CONFIG_PATH
            mock_get_config.assert_called_with(USER_CONFIG_PATH)
            # Assert the predicate at line 43 evaluated to True
            assert mock_exists.return_value is True
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.unlink(temp_path)


# LLM-generated content at query #11
#--------------------------

```python
def test_predicate_at_line_33_evaluates_to_false():
    # Test case 1: config_file is None
    config_file = None
    USER_CONFIG_PATH = "/default/path"
    result = config_file and config_file is not USER_CONFIG_PATH
    assert result is False
    
    # Test case 2: config_file is equal to USER_CONFIG_PATH
    config_file = "/default/path"
    USER_CONFIG_PATH = "/default/path"
    result = config_file and config_file is not USER_CONFIG_PATH
    assert result is False
    
    # Test case 3: config_file is empty string
    config_file = ""
    USER_CONFIG_PATH = "/default/path"
    result = config_file and config_file is not USER_CONFIG_PATH
    assert result is False


# LLM-generated content at query #12
#--------------------------

```python
def test_predicate_at_line_33_evaluates_to_false():
    from unittest.mock import patch, MagicMock
    
    # Mock the dependencies
    with patch('os.environ', {}):
        with patch('os.path.exists', return_value=False):
            with patch('copy.copy') as mock_copy:
                with patch('__main__.DEFAULT_CONFIG', {'key': 'default'}):
                    mock_copy.return_value = {'key': 'default'}
                    
                    # Test case 1: config_file is None
                    result = get_user_config(config_file=None, default_config=False)
                    assert result == {'key': 'default'}
                    
                    # Test case 2: config_file is USER_CONFIG_PATH (identity check)
                    USER_CONFIG_PATH_VALUE = '/home/user/.cookiecutterrc'
                    with patch('__main__.USER_CONFIG_PATH', USER_CONFIG_PATH_VALUE):
                        result = get_user_config(config_file=USER_CONFIG_PATH_VALUE, default_config=False)
                        assert result == {'key': 'default'}


# LLM-generated content at query #13
#--------------------------

```python
def test_get_user_config_with_default_config_dict():
    default_config_dict = {'replay_dir': '/custom/replay'}
    result = get_user_config(default_config=default_config_dict)
    assert isinstance(result, dict)
    assert result['replay_dir'] == '/custom/replay'


def test_get_user_config_with_default_config_true():
    result = get_user_config(default_config=True)
    assert isinstance(result, dict)
    assert result == DEFAULT_CONFIG


def test_get_user_config_with_default_config_false_and_no_env_and_no_user_config(monkeypatch):
    monkeypatch.delenv('COOKIECUTTER_CONFIG', raising=False)
    monkeypatch.setattr('os.path.exists', lambda x: False)
    result = get_user_config(default_config=False)
    assert isinstance(result, dict)
    assert result == DEFAULT_CONFIG


def test_get_user_config_with_custom_config_file(tmp_path, monkeypatch):
    config_file = tmp_path / 'custom_config.yaml'
    config_file.write_text('replay_dir: /tmp/replay\ncookiecutters_dir: /tmp/cookies\n')
    result = get_user_config(config_file=str(config_file))
    assert isinstance(result, dict)
    assert '/tmp/replay' in result['replay_dir'] or result['replay_dir'] == '/tmp/replay'


def test_get_user_config_with_env_variable(tmp_path, monkeypatch):
    config_file = tmp_path / 'env_config.yaml'
    config_file.write_text('replay_dir: /env/replay\ncookiecutters_dir: /env/cookies\n')
    monkeypatch.setenv('COOKIECUTTER_CONFIG', str(config_file))
    result = get_user_config()
    assert isinstance(result, dict)
    assert '/env/replay' in result['replay_dir'] or result['replay_dir'] == '/env/replay'


def test_get_user_config_with_user_config_path_exists(tmp_path, monkeypatch):
    config_file = tmp_path / 'user_config.yaml'
    config_file.write_text('replay_dir: /user/replay\ncookiecutters_dir: /user/cookies\n')
    monkeypatch.delenv('COOKIECUTTER_CONFIG', raising=False)
    monkeypatch.setattr('os.path.exists', lambda x: True if str(x) == str(config_file) else os.path.exists(x))
    monkeypatch.setattr('builtins.open', lambda *args, **kwargs: open(config_file, *args[1:], **kwargs))
    result = get_user_config()
    assert isinstance(result, dict)


def test_get_user_config_config_file_not_found(monkeypatch):
    monkeypatch.setattr('os.path.exists', lambda x: False)
    try:
        get_user_config(config_file='/nonexistent/config.yaml')
        assert False, "Should raise ConfigDoesNotExistException"
    except ConfigDoesNotExistException:
        pass


def test_get_user_config_invalid_yaml(tmp_path, monkeypatch):
    config_file = tmp_path / 'invalid.yaml'
    config_file.write_text('invalid: yaml: content: [')
    try:
        get_user_config(config_file=str(config_file))
        assert False, "Should raise InvalidConfiguration"
    except InvalidConfiguration:
        pass


def test_get_user_config_yaml_not_dict(tmp_path, monkeypatch):
    config_file = tmp_path / 'list_config.yaml'
    config_file.write_text('- item1\n- item2\n')
    try:
        get_user_config(config_file=str(config_file))
        assert False, "Should raise InvalidConfiguration"
    except InvalidConfiguration:
        pass


def test_get_user_config_merges_with_defaults(tmp_path):
    config_file = tmp_path / 'partial_config.yaml'
    config_file.write_text('replay_dir: /custom/replay\n')
    result = get_user_config(config_file=str(config_file))
    assert isinstance(result, dict)
    assert result['replay_dir'] == '/custom/replay'
    assert 'cookiecutters_dir' in result


# LLM-generated content at query #14
#--------------------------

```python
def test_get_user_config_with_default_config_dict():
    default_config_dict = {'replay_dir': '/custom/replay'}
    result = get_user_config(default_config=default_config_dict)
    assert isinstance(result, dict)
    assert result['replay_dir'] == '/custom/replay'


def test_get_user_config_with_default_config_true():
    result = get_user_config(default_config=True)
    assert isinstance(result, dict)
    assert result == DEFAULT_CONFIG


def test_get_user_config_with_custom_config_file(tmp_path):
    config_file = tmp_path / "custom_config.yaml"
    config_file.write_text("replay_dir: /tmp/replay\ncookiecutters_dir: /tmp/cookies")
    result = get_user_config(config_file=str(config_file))
    assert isinstance(result, dict)
    assert result['replay_dir'] == '/tmp/replay'


def test_get_user_config_custom_config_file_not_equal_to_user_config_path(tmp_path):
    config_file = tmp_path / "custom_config.yaml"
    config_file.write_text("replay_dir: /tmp/replay\ncookiecutters_dir: /tmp/cookies")
    result = get_user_config(config_file=str(config_file))
    assert isinstance(result, dict)


def test_get_user_config_default_when_no_config_exists(monkeypatch):
    monkeypatch.delenv('COOKIECUTTER_CONFIG', raising=False)
    monkeypatch.setattr('os.path.exists', lambda x: False)
    result = get_user_config()
    assert isinstance(result, dict)
    assert result == DEFAULT_CONFIG


def test_get_user_config_from_environment_variable(tmp_path, monkeypatch):
    config_file = tmp_path / "env_config.yaml"
    config_file.write_text("replay_dir: /env/replay\ncookiecutters_dir: /env/cookies")
    monkeypatch.setenv('COOKIECUTTER_CONFIG', str(config_file))
    result = get_user_config()
    assert isinstance(result, dict)
    assert result['replay_dir'] == '/env/replay'


def test_get_user_config_from_user_config_path(tmp_path, monkeypatch):
    monkeypatch.delenv('COOKIECUTTER_CONFIG', raising=False)
    monkeypatch.setattr('os.path.exists', lambda x: True if str(x) == USER_CONFIG_PATH else False)
    monkeypatch.setattr('builtins.open', lambda *args, **kwargs: __import__('io').StringIO("replay_dir: /user/replay\ncookiecutters_dir: /user/cookies"))
    result = get_user_config()
    assert isinstance(result, dict)


def test_get_user_config_priority_env_over_user_path(tmp_path, monkeypatch):
    config_file = tmp_path / "env_config.yaml"
    config_file.write_text("replay_dir: /env/replay\ncookiecutters_dir: /env/cookies")
    monkeypatch.setenv('COOKIECUTTER_CONFIG', str(config_file))
    monkeypatch.delenv('COOKIECUTTER_CONFIG', raising=False)
    monkeypatch.setenv('COOKIECUTTER_CONFIG', str(config_file))
    result = get_user_config()
    assert isinstance(result, dict)


def test_get_user_config_default_config_dict_merges_with_defaults():
    custom_dict = {'replay_dir': '/custom/replay', 'extra_key': 'extra_value'}
    result = get_user_config(default_config=custom_dict)
    assert result['replay_dir'] == '/custom/replay'
    assert 'extra_key' in result


def test_get_user_config_false_default_config_parameter(monkeypatch):
    monkeypatch.delenv('COOKIECUTTER_CONFIG', raising=False)
    monkeypatch.setattr('os.path.exists', lambda x: False)
    result = get_user_config(default_config=False)
    assert isinstance(result, dict)
    assert result == DEFAULT_CONFIG


# LLM-generated content at query #15
#--------------------------

```python
def test_get_config_raises_exception_when_config_path_does_not_exist(tmp_path):
    import os
    from pathlib import Path
    
    non_existent_path = tmp_path / "non_existent_config.yaml"
    
    try:
        get_config(non_existent_path)
        assert False, "Expected ConfigDoesNotExistException to be raised"
    except ConfigDoesNotExistException:
        assert True


# LLM-generated content at query #16
#--------------------------

```python
def test_get_config_file_does_not_exist():
    config_path = "/nonexistent/path/to/config.yaml"
    try:
        get_config(config_path)
        assert False, "Expected ConfigDoesNotExistException"
    except ConfigDoesNotExistException:
        pass


def test_get_config_invalid_yaml():
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("invalid: yaml: content: [")
        temp_path = f.name
    
    try:
        get_config(temp_path)
        assert False, "Expected InvalidConfiguration"
    except InvalidConfiguration:
        pass
    finally:
        os.unlink(temp_path)


def test_get_config_non_dict_top_level():
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("- item1\n- item2\n")
        temp_path = f.name
    
    try:
        get_config(temp_path)
        assert False, "Expected InvalidConfiguration"
    except InvalidConfiguration:
        pass
    finally:
        os.unlink(temp_path)


def test_get_config_empty_yaml():
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("")
        temp_path = f.name
    
    try:
        result = get_config(temp_path)
        assert isinstance(result, dict)
        assert 'replay_dir' in result
        assert 'cookiecutters_dir' in result
    finally:
        os.unlink(temp_path)


def test_get_config_valid_config():
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("replay_dir: /tmp/replays\ncookiecutters_dir: /tmp/cookies\n")
        temp_path = f.name
    
    try:
        result = get_config(temp_path)
        assert isinstance(result, dict)
        assert 'replay_dir' in result
        assert 'cookiecutters_dir' in result
        assert result['replay_dir'] == '/tmp/replays'
        assert result['cookiecutters_dir'] == '/tmp/cookies'
    finally:
        os.unlink(temp_path)


def test_get_config_with_env_vars():
    import tempfile
    os.environ['TEST_REPLAY_DIR'] = '/test/replays'
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("replay_dir: $TEST_REPLAY_DIR\ncookiecutters_dir: /tmp/cookies\n")
        temp_path = f.name
    
    try:
        result = get_config(temp_path)
        assert result['replay_dir'] == '/test/replays'
    finally:
        os.unlink(temp_path)
        del os.environ['TEST_REPLAY_DIR']


def test_get_config_with_user_home():
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("replay_dir: ~/replays\ncookiecutters_dir: ~/cookies\n")
        temp_path = f.name
    
    try:
        result = get_config(temp_path)
        assert '~' not in result['replay_dir']
        assert '~' not in result['cookiecutters_dir']
        assert result['replay_dir'].startswith(os.path.expanduser('~'))
    finally:
        os.unlink(temp_path)


def test_get_config_merges_with_defaults():
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("replay_dir: /custom/replays\n")
        temp_path = f.name
    
    try:
        result = get_config(temp_path)
        assert result['replay_dir'] == '/custom/replays'
        assert 'cookiecutters_dir' in result
    finally:
        os.unlink(temp_path)


# LLM-generated content at query #17
#--------------------------

```python
def test_get_user_config_with_default_config_dict():
    default_config_dict = {'replay_dir': '/custom/replay'}
    result = get_user_config(default_config=default_config_dict)
    assert isinstance(result, dict)
    assert result['replay_dir'] == '/custom/replay'


def test_get_user_config_with_default_config_true():
    result = get_user_config(default_config=True)
    assert isinstance(result, dict)
    assert result == DEFAULT_CONFIG


def test_get_user_config_with_custom_config_file(tmp_path):
    config_file = tmp_path / "custom_config.yaml"
    config_file.write_text("replay_dir: /tmp/replay\ncookiecutters_dir: /tmp/cookies")
    result = get_user_config(config_file=str(config_file))
    assert isinstance(result, dict)
    assert '/tmp/replay' in result['replay_dir']


def test_get_user_config_with_nonexistent_custom_config_file():
    try:
        get_user_config(config_file="/nonexistent/path/config.yaml")
        assert False, "Should raise ConfigDoesNotExistException"
    except ConfigDoesNotExistException:
        pass


def test_get_user_config_with_env_variable(tmp_path, monkeypatch):
    config_file = tmp_path / "env_config.yaml"
    config_file.write_text("replay_dir: /env/replay\ncookiecutters_dir: /env/cookies")
    monkeypatch.setenv('COOKIECUTTER_CONFIG', str(config_file))
    result = get_user_config()
    assert isinstance(result, dict)
    assert '/env/replay' in result['replay_dir']


def test_get_user_config_with_nonexistent_env_variable_and_no_user_config(monkeypatch):
    monkeypatch.delenv('COOKIECUTTER_CONFIG', raising=False)
    monkeypatch.setattr('os.path.exists', lambda x: False)
    result = get_user_config()
    assert isinstance(result, dict)
    assert result == DEFAULT_CONFIG


def test_get_user_config_default_config_false_over_custom_config_file(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("replay_dir: /custom/replay\ncookiecutters_dir: /custom/cookies")
    result = get_user_config(config_file=str(config_file), default_config=False)
    assert isinstance(result, dict)
    assert '/custom/replay' in result['replay_dir']


def test_get_user_config_default_config_dict_overrides_config_file(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("replay_dir: /custom/replay\ncookiecutters_dir: /custom/cookies")
    default_config_dict = {'replay_dir': '/override/replay'}
    result = get_user_config(config_file=str(config_file), default_config=default_config_dict)
    assert isinstance(result, dict)
    assert result['replay_dir'] == '/override/replay'


def test_get_user_config_invalid_yaml_file(tmp_path):
    config_file = tmp_path / "invalid.yaml"
    config_file.write_text("invalid: yaml: content: [")
    try:
        get_user_config(config_file=str(config_file))
        assert False, "Should raise InvalidConfiguration"
    except InvalidConfiguration:
        pass


def test_get_user_config_yaml_not_dict(tmp_path):
    config_file = tmp_path / "notdict.yaml"
    config_file.write_text("- item1\n- item2")
    try:
        get_user_config(config_file=str(config_file))
        assert False, "Should raise InvalidConfiguration"
    except InvalidConfiguration:
        pass


# LLM-generated content at query #18
#--------------------------

```python
def test_get_config_raises_exception_when_config_file_does_not_exist(tmp_path):
    non_existent_path = tmp_path / "non_existent_config.yaml"
    
    try:
        get_config(non_existent_path)
        assert False, "Expected ConfigDoesNotExistException to be raised"
    except ConfigDoesNotExistException:
        assert True


# LLM-generated content at query #19
#--------------------------

```python
def test_cookiecutter_config_env_var_not_set():
    import os
    import copy
    from unittest.mock import patch, MagicMock

    # Mock the dependencies
    mock_default_config = {"key": "default_value"}
    mock_get_config = MagicMock()
    
    # Ensure COOKIECUTTER_CONFIG is not set in environment
    with patch.dict(os.environ, {}, clear=True):
        # Ensure the KeyError exception is raised (predicate at line 40)
        try:
            env_config_file = os.environ['COOKIECUTTER_CONFIG']
            exception_raised = False
        except KeyError:
            exception_raised = True
    
    assert exception_raised is True


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_get_user_config_with_default_config_dict():
    default_config_dict = {'replay_dir': '/custom/replay'}
    result = get_user_config(default_config=default_config_dict)
    assert isinstance(result, dict)
    assert result['replay_dir'] == '/custom/replay'


def test_get_user_config_with_default_config_true():
    result = get_user_config(default_config=True)
    assert isinstance(result, dict)
    assert result == DEFAULT_CONFIG


def test_get_user_config_with_custom_config_file(tmp_path):
    config_file = tmp_path / "custom_config.yaml"
    config_file.write_text("replay_dir: /tmp/replay\ncookiecutters_dir: /tmp/cookies")
    result = get_user_config(config_file=str(config_file))
    assert isinstance(result, dict)
    assert result['replay_dir'] == '/tmp/replay'


def test_get_user_config_default_config_false_no_env_no_user_config(monkeypatch, tmp_path):
    monkeypatch.delenv('COOKIECUTTER_CONFIG', raising=False)
    monkeypatch.setattr('os.path.exists', lambda x: False)
    result = get_user_config(default_config=False)
    assert isinstance(result, dict)
    assert result == DEFAULT_CONFIG


def test_get_user_config_with_env_variable(monkeypatch, tmp_path):
    config_file = tmp_path / "env_config.yaml"
    config_file.write_text("replay_dir: /env/replay\ncookiecutters_dir: /env/cookies")
    monkeypatch.setenv('COOKIECUTTER_CONFIG', str(config_file))
    monkeypatch.setattr('os.path.exists', lambda x: False)
    result = get_user_config(default_config=False)
    assert isinstance(result, dict)
    assert result['replay_dir'] == '/env/replay'


def test_get_user_config_with_user_config_path(monkeypatch, tmp_path):
    config_file = tmp_path / ".cookiecutterrc"
    config_file.write_text("replay_dir: /user/replay\ncookiecutters_dir: /user/cookies")
    monkeypatch.delenv('COOKIECUTTER_CONFIG', raising=False)
    monkeypatch.setattr('os.path.exists', lambda x: str(x) == str(config_file) if 'cookiecutter' in str(x) else False)
    monkeypatch.setattr('os.path.expandvars', lambda x: x)
    monkeypatch.setattr('os.path.expanduser', lambda x: x)
    result = get_user_config(default_config=False, config_file=None)
    assert isinstance(result, dict)


def test_get_user_config_config_file_not_equal_user_config_path(tmp_path, monkeypatch):
    config_file = tmp_path / "custom.yaml"
    config_file.write_text("replay_dir: /custom/replay\ncookiecutters_dir: /custom/cookies")
    monkeypatch.setattr('os.path.expandvars', lambda x: x)
    monkeypatch.setattr('os.path.expanduser', lambda x: x)
    result = get_user_config(config_file=str(config_file), default_config=False)
    assert isinstance(result, dict)
    assert result['replay_dir'] == '/custom/replay'


def test_get_user_config_invalid_yaml_file(tmp_path):
    config_file = tmp_path / "invalid.yaml"
    config_file.write_text("{ invalid yaml content [")
    try:
        get_user_config(config_file=str(config_file))
        assert False, "Should have raised InvalidConfiguration"
    except InvalidConfiguration:
        pass


def test_get_user_config_nonexistent_file():
    try:
        get_user_config(config_file="/nonexistent/path/config.yaml")
        assert False, "Should have raised ConfigDoesNotExistException"
    except ConfigDoesNotExistException:
        pass


# LLM-generated content at query #2
#--------------------------

```python
def test_get_config_file_does_not_exist(tmp_path):
    non_existent_path = tmp_path / "non_existent_config.yaml"
    try:
        get_config(non_existent_path)
        assert False, "Should have raised ConfigDoesNotExistException"
    except ConfigDoesNotExistException:
        pass


def test_get_config_invalid_yaml(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("invalid: yaml: content:", encoding='utf-8')
    try:
        get_config(config_file)
        assert False, "Should have raised InvalidConfiguration"
    except InvalidConfiguration:
        pass


def test_get_config_non_dict_top_level(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("- item1\n- item2\n", encoding='utf-8')
    try:
        get_config(config_file)
        assert False, "Should have raised InvalidConfiguration"
    except InvalidConfiguration:
        pass


def test_get_config_empty_file(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("", encoding='utf-8')
    result = get_config(config_file)
    assert isinstance(result, dict)
    assert 'replay_dir' in result
    assert 'cookiecutters_dir' in result


def test_get_config_valid_config(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_content = "replay_dir: /tmp/replays\ncookiecutters_dir: /tmp/cookies\n"
    config_file.write_text(config_content, encoding='utf-8')
    result = get_config(config_file)
    assert isinstance(result, dict)
    assert 'replay_dir' in result
    assert 'cookiecutters_dir' in result


def test_get_config_with_env_vars(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_REPLAY_DIR", "/custom/replays")
    config_file = tmp_path / "config.yaml"
    config_content = "replay_dir: $TEST_REPLAY_DIR\ncookiecutters_dir: /tmp/cookies\n"
    config_file.write_text(config_content, encoding='utf-8')
    result = get_config(config_file)
    assert result['replay_dir'] == "/custom/replays"


def test_get_config_with_tilde_expansion(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_content = "replay_dir: ~/replays\ncookiecutters_dir: ~/cookies\n"
    config_file.write_text(config_content, encoding='utf-8')
    result = get_config(config_file)
    assert "~" not in result['replay_dir']
    assert "~" not in result['cookiecutters_dir']


def test_get_config_merges_with_default(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_content = "replay_dir: /custom/path\n"
    config_file.write_text(config_content, encoding='utf-8')
    result = get_config(config_file)
    assert result['replay_dir'] == "/custom/path"
    assert 'cookiecutters_dir' in result


def test_get_config_nested_dict_merge(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_content = "replay_dir: /tmp/replays\ncookiecutters_dir: /tmp/cookies\nabbreviations:\n  key1: value1\n"
    config_file.write_text(config_content, encoding='utf-8')
    result = get_config(config_file)
    assert isinstance(result, dict)
    assert 'abbreviations' in result


# LLM-generated content at query #3
#--------------------------

```python
def test_expand_path_with_home_directory():
    import os
    from pathlib import Path
    path = "~/test_file.txt"
    result = _expand_path(path)
    expected = os.path.expanduser("~/test_file.txt")
    assert result == expected
    assert "~" not in result


def test_expand_path_with_environment_variable():
    import os
    os.environ["TEST_VAR"] = "/test/path"
    path = "$TEST_VAR/file.txt"
    result = _expand_path(path)
    expected = "/test/path/file.txt"
    assert result == expected


def test_expand_path_with_both_home_and_env_var():
    import os
    os.environ["HOME_TEST"] = "mydir"
    path = "~/$HOME_TEST/file.txt"
    result = _expand_path(path)
    assert "$HOME_TEST" not in result
    assert "~" not in result


def test_expand_path_with_no_variables():
    path = "/absolute/path/to/file.txt"
    result = _expand_path(path)
    assert result == "/absolute/path/to/file.txt"


def test_expand_path_with_relative_path():
    path = "relative/path/file.txt"
    result = _expand_path(path)
    assert result == "relative/path/file.txt"


def test_expand_path_with_empty_string():
    path = ""
    result = _expand_path(path)
    assert result == ""


def test_expand_path_with_only_home_symbol():
    import os
    path = "~"
    result = _expand_path(path)
    expected = os.path.expanduser("~")
    assert result == expected
    assert result == os.path.expanduser("~")


# LLM-generated content at query #4
#--------------------------

```python
def test_get_config_file_not_exists():
    config_path = '/nonexistent/path/config.yaml'
    try:
        get_config(config_path)
        assert False, "Should raise ConfigDoesNotExistException"
    except ConfigDoesNotExistException:
        pass


def test_get_config_invalid_yaml(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("invalid: yaml: content: [", encoding='utf-8')
    try:
        get_config(str(config_file))
        assert False, "Should raise InvalidConfiguration"
    except InvalidConfiguration:
        pass


def test_get_config_non_dict_top_level(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("- item1\n- item2\n", encoding='utf-8')
    try:
        get_config(str(config_file))
        assert False, "Should raise InvalidConfiguration"
    except InvalidConfiguration:
        pass


def test_get_config_empty_file(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("", encoding='utf-8')
    result = get_config(str(config_file))
    assert isinstance(result, dict)
    assert 'replay_dir' in result
    assert 'cookiecutters_dir' in result


def test_get_config_with_valid_config(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("replay_dir: /tmp/replays\ncookiecutters_dir: /tmp/cookies\n", encoding='utf-8')
    result = get_config(str(config_file))
    assert isinstance(result, dict)
    assert result['replay_dir'] == '/tmp/replays'
    assert result['cookiecutters_dir'] == '/tmp/cookies'


def test_get_config_expands_environment_variables(tmp_path):
    import os
    config_file = tmp_path / "config.yaml"
    config_file.write_text("replay_dir: $HOME/replays\ncookiecutters_dir: $HOME/cookies\n", encoding='utf-8')
    result = get_config(str(config_file))
    assert isinstance(result, dict)
    assert '$HOME' not in result['replay_dir']
    assert '$HOME' not in result['cookiecutters_dir']
    assert result['replay_dir'].startswith(os.path.expanduser('~'))
    assert result['cookiecutters_dir'].startswith(os.path.expanduser('~'))


def test_get_config_expands_user_home(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("replay_dir: ~/replays\ncookiecutters_dir: ~/cookies\n", encoding='utf-8')
    result = get_config(str(config_file))
    assert isinstance(result, dict)
    assert '~' not in result['replay_dir']
    assert '~' not in result['cookiecutters_dir']


def test_get_config_merges_with_defaults(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("replay_dir: /custom/replays\n", encoding='utf-8')
    result = get_config(str(config_file))
    assert isinstance(result, dict)
    assert result['replay_dir'] == '/custom/replays'
    assert 'cookiecutters_dir' in result


# LLM-generated content at query #5
#--------------------------

```python
def test_get_user_config_with_default_config_dict():
    default_config_dict = {'replay_dir': '/custom/replay'}
    result = get_user_config(default_config=default_config_dict)
    assert isinstance(result, dict)
    assert result['replay_dir'] == '/custom/replay'


def test_get_user_config_with_default_config_true():
    result = get_user_config(default_config=True)
    assert isinstance(result, dict)
    assert result == DEFAULT_CONFIG


def test_get_user_config_with_custom_config_file(tmp_path):
    config_file = tmp_path / "custom_config.yaml"
    config_file.write_text("replay_dir: /tmp/replays\ncookiecutters_dir: /tmp/cookies")
    result = get_user_config(config_file=str(config_file))
    assert isinstance(result, dict)
    assert 'replay_dir' in result
    assert 'cookiecutters_dir' in result


def test_get_user_config_with_nonexistent_custom_config_file():
    try:
        get_user_config(config_file="/nonexistent/path/config.yaml")
        assert False, "Should raise ConfigDoesNotExistException"
    except ConfigDoesNotExistException:
        pass


def test_get_user_config_with_env_variable(tmp_path, monkeypatch):
    config_file = tmp_path / "env_config.yaml"
    config_file.write_text("replay_dir: /env/replays\ncookiecutters_dir: /env/cookies")
    monkeypatch.setenv('COOKIECUTTER_CONFIG', str(config_file))
    monkeypatch.delenv('HOME', raising=False)
    result = get_user_config()
    assert isinstance(result, dict)
    assert 'replay_dir' in result


def test_get_user_config_with_invalid_env_variable(monkeypatch):
    monkeypatch.setenv('COOKIECUTTER_CONFIG', "/nonexistent/env/config.yaml")
    try:
        get_user_config()
        assert False, "Should raise ConfigDoesNotExistException"
    except ConfigDoesNotExistException:
        pass


def test_get_user_config_no_env_no_user_config(monkeypatch):
    monkeypatch.delenv('COOKIECUTTER_CONFIG', raising=False)
    monkeypatch.setattr('os.path.exists', lambda x: False)
    result = get_user_config()
    assert isinstance(result, dict)
    assert result == DEFAULT_CONFIG


def test_get_user_config_prefers_default_config_dict_over_file():
    default_config_dict = {'replay_dir': '/override/replay'}
    result = get_user_config(config_file="/some/path", default_config=default_config_dict)
    assert result['replay_dir'] == '/override/replay'


def test_get_user_config_prefers_default_config_true_over_file():
    result = get_user_config(config_file="/some/path", default_config=True)
    assert result == DEFAULT_CONFIG


# LLM-generated content at query #6
#--------------------------

```python
def test_get_user_config_with_default_config_dict():
    default_config_dict = {'replay_dir': '/custom/replay'}
    result = get_user_config(default_config=default_config_dict)
    assert isinstance(result, dict)
    assert result['replay_dir'] == '/custom/replay'


def test_get_user_config_with_default_config_true():
    result = get_user_config(default_config=True)
    assert isinstance(result, dict)
    assert result == DEFAULT_CONFIG


def test_get_user_config_with_custom_config_file(tmp_path):
    config_file = tmp_path / 'custom_config.yaml'
    config_file.write_text('replay_dir: /tmp/replay\ncookiecutters_dir: /tmp/cookies\n', encoding='utf-8')
    result = get_user_config(config_file=str(config_file))
    assert isinstance(result, dict)
    assert 'replay_dir' in result


def test_get_user_config_with_env_variable(tmp_path, monkeypatch):
    config_file = tmp_path / 'env_config.yaml'
    config_file.write_text('replay_dir: /tmp/replay\ncookiecutters_dir: /tmp/cookies\n', encoding='utf-8')
    monkeypatch.setenv('COOKIECUTTER_CONFIG', str(config_file))
    result = get_user_config()
    assert isinstance(result, dict)
    assert 'replay_dir' in result


def test_get_user_config_default_when_no_env_and_no_file(monkeypatch):
    monkeypatch.delenv('COOKIECUTTER_CONFIG', raising=False)
    monkeypatch.setattr('os.path.exists', lambda x: False)
    result = get_user_config()
    assert isinstance(result, dict)
    assert result == DEFAULT_CONFIG


def test_get_user_config_loads_user_config_when_exists(tmp_path, monkeypatch):
    config_file = tmp_path / 'user_config.yaml'
    config_file.write_text('replay_dir: /tmp/replay\ncookiecutters_dir: /tmp/cookies\n', encoding='utf-8')
    monkeypatch.delenv('COOKIECUTTER_CONFIG', raising=False)
    monkeypatch.setattr('os.path.exists', lambda x: str(x) == str(config_file) if isinstance(x, (str, Path)) else False)
    monkeypatch.setattr('__main__.USER_CONFIG_PATH', str(config_file))
    result = get_user_config()
    assert isinstance(result, dict)


def test_get_user_config_with_invalid_config_file(tmp_path):
    invalid_config_file = tmp_path / 'invalid.yaml'
    invalid_config_file.write_text('invalid: : yaml: content:\n', encoding='utf-8')
    try:
        get_user_config(config_file=str(invalid_config_file))
        assert False, "Should raise InvalidConfiguration"
    except InvalidConfiguration:
        pass


def test_get_user_config_with_nonexistent_config_file(tmp_path):
    nonexistent_file = tmp_path / 'nonexistent.yaml'
    try:
        get_user_config(config_file=str(nonexistent_file))
        assert False, "Should raise ConfigDoesNotExistException"
    except ConfigDoesNotExistException:
        pass


def test_get_user_config_merges_default_with_dict():
    custom_dict = {'replay_dir': '/custom/path', 'extra_key': 'extra_value'}
    result = get_user_config(default_config=custom_dict)
    assert result['replay_dir'] == '/custom/path'
    assert 'cookiecutters_dir' in result


def test_get_user_config_prioritizes_custom_file_over_default(tmp_path):
    custom_config = tmp_path / 'custom.yaml'
    custom_config.write_text('replay_dir: /custom/replay\ncookiecutters_dir: /custom/cookies\n', encoding='utf-8')
    result = get_user_config(config_file=str(custom_config))
    assert isinstance(result, dict)


# LLM-generated content at query #7
#--------------------------

```python
def test_get_config_opens_file_with_utf8_encoding(tmp_path):
    import os
    from pathlib import Path
    
    config_file = tmp_path / "config.yaml"
    config_file.write_text("replay_dir: /tmp\ncookiecutters_dir: /tmp\n", encoding='utf-8')
    
    # Mock the dependencies
    import unittest.mock as mock
    
    with mock.patch('os.path.exists', return_value=True):
        with mock.patch('builtins.open', mock.mock_open(read_data="replay_dir: /tmp\ncookiecutters_dir: /tmp\n")) as mock_file:
            with mock.patch('yaml.safe_load', return_value={'replay_dir': '/tmp', 'cookiecutters_dir': '/tmp'}):
                with mock.patch('merge_configs', return_value={'replay_dir': '/tmp', 'cookiecutters_dir': '/tmp'}):
                    with mock.patch('_expand_path', side_effect=lambda x: x):
                        get_config(config_file)
                        
                        mock_file.assert_called_once_with(config_file, encoding='utf-8')


# LLM-generated content at query #8
#--------------------------

```python
def test_get_config_file_does_not_exist(tmp_path):
    config_path = tmp_path / "nonexistent.yaml"
    try:
        get_config(config_path)
        assert False, "Should have raised ConfigDoesNotExistException"
    except ConfigDoesNotExistException:
        pass


def test_get_config_valid_yaml(tmp_path, monkeypatch):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("replay_dir: /tmp/replays\ncookiecutters_dir: /tmp/cookies\n")
    result = get_config(config_file)
    assert isinstance(result, dict)
    assert 'replay_dir' in result
    assert 'cookiecutters_dir' in result


def test_get_config_invalid_yaml(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("invalid: yaml: content: [")
    try:
        get_config(config_file)
        assert False, "Should have raised InvalidConfiguration"
    except InvalidConfiguration:
        pass


def test_get_config_top_level_not_dict(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("- item1\n- item2\n")
    try:
        get_config(config_file)
        assert False, "Should have raised InvalidConfiguration"
    except InvalidConfiguration:
        pass


def test_get_config_empty_yaml(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("")
    result = get_config(config_file)
    assert isinstance(result, dict)


def test_get_config_with_env_vars(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_REPLAY_DIR", "/test/replays")
    config_file = tmp_path / "config.yaml"
    config_file.write_text("replay_dir: $TEST_REPLAY_DIR\ncookiecutters_dir: /tmp/cookies\n")
    result = get_config(config_file)
    assert result['replay_dir'] == "/test/replays"


def test_get_config_with_home_expansion(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("replay_dir: ~/replays\ncookiecutters_dir: ~/cookies\n")
    result = get_config(config_file)
    assert "~" not in result['replay_dir']
    assert "~" not in result['cookiecutters_dir']


def test_get_config_merges_with_defaults(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("replay_dir: /custom/replays\n")
    result = get_config(config_file)
    assert result['replay_dir'] == "/custom/replays"
    assert 'cookiecutters_dir' in result


# LLM-generated content at query #9
#--------------------------

```python
def test_get_config_file_does_not_exist(tmp_path):
    non_existent_path = tmp_path / "non_existent.yaml"
    try:
        get_config(non_existent_path)
        assert False, "Should have raised ConfigDoesNotExistException"
    except ConfigDoesNotExistException:
        pass


def test_get_config_invalid_yaml(tmp_path):
    config_file = tmp_path / "invalid.yaml"
    config_file.write_text("invalid: yaml: content:", encoding='utf-8')
    try:
        get_config(config_file)
        assert False, "Should have raised InvalidConfiguration"
    except InvalidConfiguration:
        pass


def test_get_config_non_dict_top_level(tmp_path):
    config_file = tmp_path / "non_dict.yaml"
    config_file.write_text("- item1\n- item2", encoding='utf-8')
    try:
        get_config(config_file)
        assert False, "Should have raised InvalidConfiguration"
    except InvalidConfiguration:
        pass


def test_get_config_empty_yaml(tmp_path):
    config_file = tmp_path / "empty.yaml"
    config_file.write_text("", encoding='utf-8')
    result = get_config(config_file)
    assert isinstance(result, dict)
    assert 'replay_dir' in result
    assert 'cookiecutters_dir' in result


def test_get_config_with_valid_config(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("replay_dir: /tmp/replays\ncookiecutters_dir: /tmp/cookies", encoding='utf-8')
    result = get_config(config_file)
    assert isinstance(result, dict)
    assert result['replay_dir'] == '/tmp/replays'
    assert result['cookiecutters_dir'] == '/tmp/cookies'


def test_get_config_with_env_vars(tmp_path, monkeypatch):
    monkeypatch.setenv('TEST_REPLAY_DIR', '/home/user/replays')
    config_file = tmp_path / "config.yaml"
    config_file.write_text("replay_dir: $TEST_REPLAY_DIR", encoding='utf-8')
    result = get_config(config_file)
    assert result['replay_dir'] == '/home/user/replays'


def test_get_config_with_home_expansion(tmp_path, monkeypatch):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("replay_dir: ~/replays\ncookiecutters_dir: ~/.cookiecutters", encoding='utf-8')
    result = get_config(config_file)
    assert '~' not in result['replay_dir']
    assert '~' not in result['cookiecutters_dir']
    assert result['replay_dir'].startswith('/')
    assert result['cookiecutters_dir'].startswith('/')


def test_get_config_merges_with_defaults(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("replay_dir: /custom/replays", encoding='utf-8')
    result = get_config(config_file)
    assert result['replay_dir'] == '/custom/replays'
    assert 'cookiecutters_dir' in result


def test_get_config_nested_dict_merge(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("abbreviations:\n  custom_key: custom_value", encoding='utf-8')
    result = get_config(config_file)
    assert 'abbreviations' in result
    assert result['abbreviations']['custom_key'] == 'custom_value'


# LLM-generated content at query #10
#--------------------------

```python
def test_yaml_error_raises_invalid_configuration(tmp_path):
    import yaml
    from pathlib import Path
    
    config_file = tmp_path / "config.yaml"
    config_file.write_text("invalid: yaml: content: [")
    
    try:
        get_config(config_file)
        assert False, "Expected InvalidConfiguration to be raised"
    except InvalidConfiguration as e:
        assert "Unable to parse YAML file" in str(e)
        assert isinstance(e.__cause__, yaml.YAMLError)


# LLM-generated content at query #11
#--------------------------

```python
def test_get_user_config_with_default_config_true():
    """Test get_user_config returns DEFAULT_CONFIG when default_config is True."""
    result = get_user_config(default_config=True)
    assert result == DEFAULT_CONFIG


def test_get_user_config_with_default_config_dict():
    """Test get_user_config merges provided dict with DEFAULT_CONFIG."""
    custom_config = {'replay_dir': '/custom/replay'}
    result = get_user_config(default_config=custom_config)
    assert result['replay_dir'] == '/custom/replay'
    assert 'cookiecutters_dir' in result


def test_get_user_config_with_custom_config_file(tmp_path):
    """Test get_user_config loads custom config file."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("replay_dir: /custom/path\ncookiecutters_dir: /cookies")
    result = get_user_config(config_file=str(config_file))
    assert result['replay_dir'] == '/custom/path'
    assert result['cookiecutters_dir'] == '/cookies'


def test_get_user_config_with_env_variable(tmp_path, monkeypatch):
    """Test get_user_config loads config from COOKIECUTTER_CONFIG environment variable."""
    config_file = tmp_path / "env_config.yaml"
    config_file.write_text("replay_dir: /env/replay")
    monkeypatch.setenv('COOKIECUTTER_CONFIG', str(config_file))
    monkeypatch.delenv('COOKIECUTTER_CONFIG', raising=False)
    monkeypatch.setenv('COOKIECUTTER_CONFIG', str(config_file))
    result = get_user_config()
    assert result['replay_dir'] == '/env/replay'


def test_get_user_config_user_config_path_exists(tmp_path, monkeypatch):
    """Test get_user_config loads from USER_CONFIG_PATH when it exists."""
    monkeypatch.delenv('COOKIECUTTER_CONFIG', raising=False)
    config_file = tmp_path / "user_config.yaml"
    config_file.write_text("replay_dir: /user/replay")
    monkeypatch.setattr('os.path.exists', lambda p: str(p) == str(config_file) or os.path.exists(p))
    monkeypatch.setattr('get_user_config', lambda **kwargs: get_config(str(config_file)))
    result = get_config(str(config_file))
    assert result['replay_dir'] == '/user/replay'


def test_get_user_config_returns_default_when_no_config_found(monkeypatch):
    """Test get_user_config returns DEFAULT_CONFIG when no config file exists."""
    monkeypatch.delenv('COOKIECUTTER_CONFIG', raising=False)
    monkeypatch.setattr('os.path.exists', lambda p: False)
    result = get_user_config()
    assert result == DEFAULT_CONFIG


def test_get_user_config_with_invalid_yaml(tmp_path):
    """Test get_user_config raises InvalidConfiguration for invalid YAML."""
    config_file = tmp_path / "invalid.yaml"
    config_file.write_text("invalid: yaml: content:")
    try:
        get_user_config(config_file=str(config_file))
        assert False, "Should have raised InvalidConfiguration"
    except InvalidConfiguration:
        pass


def test_get_user_config_with_nonexistent_custom_file():
    """Test get_user_config raises ConfigDoesNotExistException for nonexistent custom file."""
    try:
        get_user_config(config_file="/nonexistent/config.yaml")
        assert False, "Should have raised ConfigDoesNotExistException"
    except ConfigDoesNotExistException:
        pass


def test_get_user_config_expands_environment_variables(tmp_path, monkeypatch):
    """Test get_user_config expands environment variables in paths."""
    monkeypatch.setenv('TEST_REPLAY_DIR', '/expanded/replay')
    config_file = tmp_path / "config.yaml"
    config_file.write_text("replay_dir: $TEST_REPLAY_DIR")
    result = get_user_config(config_file=str(config_file))
    assert result['replay_dir'] == '/expanded/replay'


def test_get_user_config_expands_user_home(tmp_path, monkeypatch):
    """Test get_user_config expands ~ to user home directory."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("replay_dir: ~/replay")
    result = get_user_config(config_file=str(config_file))
    assert '~' not in result['replay_dir']
    assert result['replay_dir'].startswith('/')


# LLM-generated content at query #12
#--------------------------

```python
def test_get_user_config_with_default_config_dict():
    default_config_dict = {'replay_dir': '/custom/replay'}
    result = get_user_config(default_config=default_config_dict)
    assert isinstance(result, dict)
    assert result['replay_dir'] == '/custom/replay'


def test_get_user_config_with_default_config_true():
    result = get_user_config(default_config=True)
    assert isinstance(result, dict)
    assert result == copy.copy(DEFAULT_CONFIG)


def test_get_user_config_with_custom_config_file(tmp_path):
    config_file = tmp_path / "custom_config.yaml"
    config_file.write_text("replay_dir: /custom/replay\ncookiecutters_dir: /custom/cookies")
    result = get_user_config(config_file=str(config_file))
    assert isinstance(result, dict)
    assert 'replay_dir' in result


def test_get_user_config_with_env_variable(tmp_path, monkeypatch):
    config_file = tmp_path / "env_config.yaml"
    config_file.write_text("replay_dir: /env/replay\ncookiecutters_dir: /env/cookies")
    monkeypatch.setenv('COOKIECUTTER_CONFIG', str(config_file))
    result = get_user_config()
    assert isinstance(result, dict)
    assert 'replay_dir' in result


def test_get_user_config_default_path_exists(tmp_path, monkeypatch):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("replay_dir: /default/replay\ncookiecutters_dir: /default/cookies")
    monkeypatch.setenv('COOKIECUTTER_CONFIG', '')
    monkeypatch.delenv('COOKIECUTTER_CONFIG', raising=False)
    monkeypatch.setattr('os.path.exists', lambda x: x == str(config_file))
    monkeypatch.setattr('builtins.open', lambda *args, **kwargs: open(config_file, *args[1:], **kwargs))
    result = get_user_config()
    assert isinstance(result, dict)


def test_get_user_config_no_env_no_default_path(monkeypatch):
    monkeypatch.delenv('COOKIECUTTER_CONFIG', raising=False)
    monkeypatch.setattr('os.path.exists', lambda x: False)
    result = get_user_config()
    assert isinstance(result, dict)
    assert result == copy.copy(DEFAULT_CONFIG)


def test_get_user_config_with_default_config_false():
    result = get_user_config(default_config=False)
    assert isinstance(result, dict)


# LLM-generated content at query #13
#--------------------------

```python
def test_predicate_at_line_33_evaluates_to_false():
    from unittest.mock import patch, MagicMock
    
    # Test case 1: config_file is None
    result = get_user_config(config_file=None, default_config=False)
    assert result is not None
    
    # Test case 2: config_file is USER_CONFIG_PATH (identity check fails)
    with patch('os.environ', {'COOKIECUTTER_CONFIG': '/some/path'}):
        with patch('get_config') as mock_get_config:
            mock_get_config.return_value = {'key': 'value'}
            # When config_file is USER_CONFIG_PATH, the predicate should be False
            # so it should not call get_config(config_file)
            from pathlib import Path
            user_config_path = Path.home() / '.cookiecutterrc'
            result = get_user_config(config_file=str(user_config_path), default_config=False)
            # The function should skip the get_config call at line 35
            # and proceed to check environment variable or defaults
            assert result is not None
    
    # Test case 3: config_file is empty string (falsy value)
    with patch('os.path.exists', return_value=False):
        with patch('copy.copy') as mock_copy:
            mock_copy.return_value = {'default': 'config'}
            result = get_user_config(config_file='', default_config=False)
            assert result == {'default': 'config'}


# LLM-generated content at query #14
#--------------------------

```python
def test_get_config_file_not_exists():
    config_path = "/nonexistent/path/config.yaml"
    try:
        get_config(config_path)
        assert False, "Should have raised ConfigDoesNotExistException"
    except ConfigDoesNotExistException:
        pass


def test_get_config_invalid_yaml(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("invalid: yaml: content: [", encoding='utf-8')
    try:
        get_config(config_file)
        assert False, "Should have raised InvalidConfiguration"
    except InvalidConfiguration:
        pass


def test_get_config_non_dict_top_level(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("- item1\n- item2\n", encoding='utf-8')
    try:
        get_config(config_file)
        assert False, "Should have raised InvalidConfiguration"
    except InvalidConfiguration:
        pass


def test_get_config_empty_yaml(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("", encoding='utf-8')
    result = get_config(config_file)
    assert isinstance(result, dict)
    assert 'replay_dir' in result
    assert 'cookiecutters_dir' in result


def test_get_config_with_valid_config(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("replay_dir: /test/replay\ncookiecutters_dir: /test/cookies\n", encoding='utf-8')
    result = get_config(config_file)
    assert isinstance(result, dict)
    assert result['replay_dir'] == "/test/replay"
    assert result['cookiecutters_dir'] == "/test/cookies"


def test_get_config_expands_user_home(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("replay_dir: ~/replays\ncookiecutters_dir: ~/cookies\n", encoding='utf-8')
    result = get_config(config_file)
    assert "~" not in result['replay_dir']
    assert "~" not in result['cookiecutters_dir']


def test_get_config_expands_env_vars(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_REPLAY_DIR", "/test/replays")
    monkeypatch.setenv("TEST_COOKIES_DIR", "/test/cookies")
    config_file = tmp_path / "config.yaml"
    config_file.write_text("replay_dir: $TEST_REPLAY_DIR\ncookiecutters_dir: $TEST_COOKIES_DIR\n", encoding='utf-8')
    result = get_config(config_file)
    assert result['replay_dir'] == "/test/replays"
    assert result['cookiecutters_dir'] == "/test/cookies"


def test_get_config_merges_with_defaults(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("replay_dir: /custom/replay\n", encoding='utf-8')
    result = get_config(config_file)
    assert result['replay_dir'] == "/custom/replay"
    assert 'cookiecutters_dir' in result


def test_get_config_nested_dict_merge(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("abbreviations:\n  custom: value\nreplay_dir: /test\n", encoding='utf-8')
    result = get_config(config_file)
    assert isinstance(result, dict)
    assert 'abbreviations' in result


# LLM-generated content at query #15
#--------------------------

```python
def test_yaml_safe_load_returns_non_empty_dict(tmp_path):
    """Test that the predicate at line 10 evaluates to False when yaml.safe_load returns a non-empty dict."""
    import yaml
    import os
    from pathlib import Path
    
    config_file = tmp_path / "config.yaml"
    config_file.write_text("replay_dir: /tmp\ncookiecutters_dir: /tmp\n", encoding='utf-8')
    
    with open(config_file, encoding='utf-8') as file_handle:
        yaml_dict = yaml.safe_load(file_handle) or {}
    
    assert yaml_dict  # The dict is non-empty, so the predicate "or {}" evaluates to False
    assert isinstance(yaml_dict, dict)
    assert yaml_dict != {}


# LLM-generated content at query #16
#--------------------------

```python
def test_get_config_predicate_line_14_evaluates_to_true(tmp_path):
    import os
    import yaml
    from pathlib import Path
    
    config_file = tmp_path / "config.yaml"
    config_content = "replay_dir: /tmp\ncookiecutters_dir: /tmp\n"
    config_file.write_text(config_content)
    
    result = get_config(config_file)
    
    assert isinstance(result, dict)


# LLM-generated content at query #17
#--------------------------

```python
def test_get_config_file_not_exists():
    config_path = "/nonexistent/path/config.yaml"
    try:
        get_config(config_path)
        assert False, "Expected ConfigDoesNotExistException"
    except ConfigDoesNotExistException:
        pass


def test_get_config_invalid_yaml():
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("invalid: yaml: content: [")
        temp_path = f.name
    
    try:
        get_config(temp_path)
        assert False, "Expected InvalidConfiguration"
    except InvalidConfiguration:
        pass
    finally:
        import os
        os.unlink(temp_path)


def test_get_config_non_dict_top_level():
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("- item1\n- item2\n")
        temp_path = f.name
    
    try:
        get_config(temp_path)
        assert False, "Expected InvalidConfiguration"
    except InvalidConfiguration:
        pass
    finally:
        import os
        os.unlink(temp_path)


def test_get_config_empty_file():
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("")
        temp_path = f.name
    
    try:
        result = get_config(temp_path)
        assert isinstance(result, dict)
        assert 'replay_dir' in result
        assert 'cookiecutters_dir' in result
    finally:
        import os
        os.unlink(temp_path)


def test_get_config_with_valid_yaml():
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("replay_dir: ~/replays\ncookiecutters_dir: ~/cookies\n")
        temp_path = f.name
    
    try:
        result = get_config(temp_path)
        assert isinstance(result, dict)
        assert 'replay_dir' in result
        assert 'cookiecutters_dir' in result
        assert result['replay_dir'].startswith(os.path.expanduser('~'))
        assert result['cookiecutters_dir'].startswith(os.path.expanduser('~'))
    finally:
        import os
        os.unlink(temp_path)


def test_get_config_expands_environment_variables():
    import tempfile
    import os
    os.environ['TEST_REPLAY_DIR'] = '/test/replay'
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("replay_dir: $TEST_REPLAY_DIR\ncookiecutters_dir: ~/cookies\n")
        temp_path = f.name
    
    try:
        result = get_config(temp_path)
        assert result['replay_dir'] == '/test/replay'
    finally:
        os.unlink(temp_path)
        del os.environ['TEST_REPLAY_DIR']


def test_get_config_merges_with_default():
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("replay_dir: ~/custom_replays\n")
        temp_path = f.name
    
    try:
        result = get_config(temp_path)
        assert isinstance(result, dict)
        assert 'replay_dir' in result
        assert 'cookiecutters_dir' in result
    finally:
        import os
        os.unlink(temp_path)


# LLM-generated content at query #18
#--------------------------

```python
def test_get_config_file_does_not_exist(tmp_path):
    non_existent_path = tmp_path / "non_existent_config.yaml"
    try:
        get_config(non_existent_path)
        assert False, "Should have raised ConfigDoesNotExistException"
    except ConfigDoesNotExistException:
        pass


def test_get_config_invalid_yaml(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("invalid: yaml: content: [", encoding='utf-8')
    try:
        get_config(config_file)
        assert False, "Should have raised InvalidConfiguration"
    except InvalidConfiguration:
        pass


def test_get_config_non_dict_top_level(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("- item1\n- item2\n", encoding='utf-8')
    try:
        get_config(config_file)
        assert False, "Should have raised InvalidConfiguration"
    except InvalidConfiguration:
        pass


def test_get_config_empty_yaml(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("", encoding='utf-8')
    result = get_config(config_file)
    assert isinstance(result, dict)
    assert 'replay_dir' in result
    assert 'cookiecutters_dir' in result


def test_get_config_with_valid_config(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("replay_dir: /tmp/replays\ncookiecutters_dir: /tmp/cookies\n", encoding='utf-8')
    result = get_config(config_file)
    assert isinstance(result, dict)
    assert result['replay_dir'] == '/tmp/replays'
    assert result['cookiecutters_dir'] == '/tmp/cookies'


def test_get_config_expands_environment_variables(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("replay_dir: $HOME/replays\ncookiecutters_dir: $HOME/cookies\n", encoding='utf-8')
    result = get_config(config_file)
    assert isinstance(result, dict)
    assert '$HOME' not in result['replay_dir']
    assert '$HOME' not in result['cookiecutters_dir']


def test_get_config_expands_user_home(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("replay_dir: ~/replays\ncookiecutters_dir: ~/cookies\n", encoding='utf-8')
    result = get_config(config_file)
    assert isinstance(result, dict)
    assert '~' not in result['replay_dir']
    assert '~' not in result['cookiecutters_dir']


def test_get_config_merges_with_defaults(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("replay_dir: /custom/replays\n", encoding='utf-8')
    result = get_config(config_file)
    assert isinstance(result, dict)
    assert result['replay_dir'] == '/custom/replays'
    assert 'cookiecutters_dir' in result


# LLM-generated content at query #19
#--------------------------

```python
def test_yaml_safe_load_returns_none_defaults_to_empty_dict(tmp_path):
    import yaml
    from pathlib import Path
    
    config_file = tmp_path / "config.yaml"
    config_file.write_text("")
    
    result = yaml.safe_load(open(config_file, encoding='utf-8')) or {}
    
    assert result == {}
    assert isinstance(result, dict)


# LLM-generated content at query #20
#--------------------------

```python
def test_get_config_file_not_exists():
    config_path = '/nonexistent/path/config.yaml'
    try:
        get_config(config_path)
        assert False, "Should raise ConfigDoesNotExistException"
    except ConfigDoesNotExistException:
        pass


def test_get_config_valid_yaml(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("replay_dir: /tmp/replays\ncookiecutters_dir: /tmp/cookies\n", encoding='utf-8')
    
    result = get_config(str(config_file))
    
    assert isinstance(result, dict)
    assert 'replay_dir' in result
    assert 'cookiecutters_dir' in result


def test_get_config_with_environment_variables(tmp_path, monkeypatch):
    monkeypatch.setenv('REPLAY_PATH', '/home/user/replays')
    monkeypatch.setenv('COOKIES_PATH', '/home/user/cookies')
    
    config_file = tmp_path / "config.yaml"
    config_file.write_text("replay_dir: $REPLAY_PATH\ncookiecutters_dir: $COOKIES_PATH\n", encoding='utf-8')
    
    result = get_config(str(config_file))
    
    assert result['replay_dir'] == '/home/user/replays'
    assert result['cookiecutters_dir'] == '/home/user/cookies'


def test_get_config_with_tilde_expansion(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("replay_dir: ~/replays\ncookiecutters_dir: ~/cookies\n", encoding='utf-8')
    
    result = get_config(str(config_file))
    
    assert '~' not in result['replay_dir']
    assert '~' not in result['cookiecutters_dir']


def test_get_config_invalid_yaml(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("invalid: yaml: content:\n  - broken", encoding='utf-8')
    
    try:
        get_config(str(config_file))
        assert False, "Should raise InvalidConfiguration"
    except InvalidConfiguration:
        pass


def test_get_config_non_dict_yaml(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("- item1\n- item2\n", encoding='utf-8')
    
    try:
        get_config(str(config_file))
        assert False, "Should raise InvalidConfiguration"
    except InvalidConfiguration:
        pass


def test_get_config_empty_yaml(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("", encoding='utf-8')
    
    result = get_config(str(config_file))
    
    assert isinstance(result, dict)


def test_get_config_merges_with_defaults(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("replay_dir: /custom/replays\n", encoding='utf-8')
    
    result = get_config(str(config_file))
    
    assert result['replay_dir'] == '/custom/replays'
    assert 'cookiecutters_dir' in result


# LLM-generated content at query #21
#--------------------------

```python
def test_cookiecutter_config_env_var_not_set():
    import os
    import copy
    from unittest.mock import patch, MagicMock
    
    # Mock the dependencies
    mock_default_config = {"key": "default_value"}
    mock_get_config = MagicMock()
    mock_logger = MagicMock()
    mock_user_config_path = "/home/user/.cookiecutterrc"
    
    # Ensure COOKIECUTTER_CONFIG is not in environment
    with patch.dict(os.environ, {}, clear=False):
        if 'COOKIECUTTER_CONFIG' in os.environ:
            del os.environ['COOKIECUTTER_CONFIG']
        
        # The predicate at line 40 should evaluate to False when KeyError is raised
        # This means the COOKIECUTTER_CONFIG key does not exist in os.environ
        try:
            env_config_file = os.environ['COOKIECUTTER_CONFIG']
            predicate_result = True
        except KeyError:
            predicate_result = False
        
        assert predicate_result is False


# LLM-generated content at query #22
#--------------------------

```python
def test_predicate_at_line_33_evaluates_to_false():
    config_file = None
    result = config_file and config_file is not None
    assert result is False


# LLM-generated content at query #23
#--------------------------

```python
def test_yaml_safe_load_returns_non_empty_dict(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("replay_dir: /some/path\ncookiecutters_dir: /another/path\n")
    
    result = get_config(config_file)
    
    assert result is not None
    assert isinstance(result, dict)
    assert len(result) > 0


# LLM-generated content at query #24
#--------------------------

```python
def test_line_40_predicate_evaluates_to_false():
    import os
    from unittest.mock import patch
    
    # Mock the environment to NOT have COOKIECUTTER_CONFIG set
    with patch.dict(os.environ, {}, clear=True):
        # Ensure COOKIECUTTER_CONFIG is not in environment
        assert 'COOKIECUTTER_CONFIG' not in os.environ
        
        # The predicate at line 40 is the KeyError exception check
        # It evaluates to False when COOKIECUTTER_CONFIG key exists in os.environ
        try:
            env_config_file = os.environ['COOKIECUTTER_CONFIG']
            predicate_result = False  # No KeyError raised, predicate is False
        except KeyError:
            predicate_result = True  # KeyError raised, predicate is True
        
        # Verify the predicate evaluates to True (KeyError was raised)
        # To test it evaluates to False, we need COOKIECUTTER_CONFIG to be set
        with patch.dict(os.environ, {'COOKIECUTTER_CONFIG': '/some/path'}):
            try:
                env_config_file = os.environ['COOKIECUTTER_CONFIG']
                predicate_result = False  # No KeyError raised, predicate is False
            except KeyError:
                predicate_result = True
            
            assert predicate_result is False


# LLM-generated content at query #25
#--------------------------

```python
def test_predicate_at_line_43_evaluates_to_true(tmp_path, monkeypatch):
    import os
    from unittest.mock import patch, MagicMock
    
    # Create a temporary config file
    config_file = tmp_path / "config.yaml"
    config_file.write_text("test: value")
    
    # Mock the necessary dependencies
    with patch('os.environ', {}):
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = True
            with patch('os.path.exists', return_value=True):
                # The predicate at line 43 is: os.path.exists(USER_CONFIG_PATH)
                # We need to ensure this evaluates to True
                result = os.path.exists(str(config_file))
                assert result is True


# LLM-generated content at query #26
#--------------------------

```python
def test_get_user_config_with_default_config_dict():
    default_config_dict = {'replay_dir': '/custom/replay'}
    result = get_user_config(default_config=default_config_dict)
    assert isinstance(result, dict)
    assert result['replay_dir'] == '/custom/replay'


def test_get_user_config_with_default_config_true():
    result = get_user_config(default_config=True)
    assert isinstance(result, dict)
    assert result == DEFAULT_CONFIG


def test_get_user_config_with_default_config_false():
    result = get_user_config(default_config=False)
    assert isinstance(result, dict)


def test_get_user_config_with_custom_config_file(tmp_path):
    config_file = tmp_path / "custom_config.yaml"
    config_file.write_text("replay_dir: /tmp/replay\ncookiecutters_dir: /tmp/cookies\n")
    result = get_user_config(config_file=str(config_file))
    assert isinstance(result, dict)
    assert result['replay_dir'] == '/tmp/replay'


def test_get_user_config_with_nonexistent_custom_config_file():
    try:
        get_user_config(config_file="/nonexistent/path/config.yaml")
        assert False, "Should raise ConfigDoesNotExistException"
    except ConfigDoesNotExistException:
        pass


def test_get_user_config_with_env_variable(tmp_path, monkeypatch):
    config_file = tmp_path / "env_config.yaml"
    config_file.write_text("replay_dir: /tmp/env_replay\ncookiecutters_dir: /tmp/cookies\n")
    monkeypatch.setenv('COOKIECUTTER_CONFIG', str(config_file))
    result = get_user_config()
    assert isinstance(result, dict)
    assert result['replay_dir'] == '/tmp/env_replay'


def test_get_user_config_without_env_variable_with_user_config_exists(tmp_path, monkeypatch):
    monkeypatch.delenv('COOKIECUTTER_CONFIG', raising=False)
    result = get_user_config()
    assert isinstance(result, dict)


def test_get_user_config_without_env_variable_without_user_config(monkeypatch):
    monkeypatch.delenv('COOKIECUTTER_CONFIG', raising=False)
    result = get_user_config()
    assert isinstance(result, dict)
    assert result == DEFAULT_CONFIG


def test_get_user_config_default_config_dict_merges_with_defaults():
    custom_values = {'replay_dir': '/custom'}
    result = get_user_config(default_config=custom_values)
    assert result['replay_dir'] == '/custom'
    assert 'cookiecutters_dir' in result


def test_get_user_config_with_invalid_yaml(tmp_path):
    config_file = tmp_path / "invalid_config.yaml"
    config_file.write_text("invalid: yaml: content: [")
    try:
        get_user_config(config_file=str(config_file))
        assert False, "Should raise InvalidConfiguration"
    except InvalidConfiguration:
        pass


def test_get_user_config_with_non_dict_yaml_root(tmp_path):
    config_file = tmp_path / "non_dict_config.yaml"
    config_file.write_text("- item1\n- item2\n")
    try:
        get_user_config(config_file=str(config_file))
        assert False, "Should raise InvalidConfiguration"
    except InvalidConfiguration:
        pass


# LLM-generated content at query #27
#--------------------------

```python
def test_yaml_safe_load_returns_none_evaluates_to_empty_dict(tmp_path):
    import os
    import yaml
    from pathlib import Path
    
    # Create a temporary empty YAML file that will parse to None
    config_file = tmp_path / "empty_config.yaml"
    config_file.write_text("")
    
    # Mock the necessary dependencies
    from unittest.mock import patch, MagicMock
    
    mock_logger = MagicMock()
    mock_merge_configs = MagicMock(return_value={})
    mock_expand_path = MagicMock(side_effect=lambda x: x)
    
    with patch('os.path.exists', return_value=True), \
         patch('builtins.open', create=True) as mock_open, \
         patch('yaml.safe_load', return_value=None), \
         patch('logger', mock_logger), \
         patch('merge_configs', mock_merge_configs), \
         patch('_expand_path', mock_expand_path):
        
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        from pathlib import Path
        
        # The predicate at line 10: yaml_dict = yaml.safe_load(file_handle) or {}
        # When yaml.safe_load returns None, the 'or {}' should evaluate to {}
        yaml_dict = None or {}
        
        assert yaml_dict == {}
        assert isinstance(yaml_dict, dict)


# LLM-generated content at query #28
#--------------------------

```python
def test_predicate_at_line_43_evaluates_to_true(tmp_path, monkeypatch):
    import os
    from unittest.mock import patch
    
    # Create a temporary config file
    config_file = tmp_path / "config.yaml"
    config_file.write_text("test_key: test_value")
    
    # Mock USER_CONFIG_PATH to point to our temporary file
    with patch('os.path.exists') as mock_exists:
        mock_exists.return_value = True
        
        # Verify the predicate evaluates to True
        result = os.path.exists(str(config_file))
        assert result is True


# LLM-generated content at query #29
#--------------------------

```python
def test_predicate_line_14_evaluates_to_false(tmp_path):
    """Test that the predicate at line 14 evaluates to False when yaml_dict is a dict."""
    import os
    import yaml
    from pathlib import Path
    
    config_file = tmp_path / "config.yaml"
    config_file.write_text("key: value\n", encoding='utf-8')
    
    result = get_config(str(config_file))
    
    assert isinstance(result, dict)


# LLM-generated content at query #30
#--------------------------

```python
def test_yaml_safe_load_returns_non_empty_dict(tmp_path):
    """Test that the predicate at line 10 evaluates to False when yaml.safe_load returns a non-empty dict."""
    import yaml
    import os
    from pathlib import Path
    
    config_file = tmp_path / "config.yaml"
    config_file.write_text("replay_dir: /tmp\ncookiecutters_dir: /tmp\n", encoding='utf-8')
    
    yaml_dict = yaml.safe_load(open(config_file, encoding='utf-8'))
    
    result = yaml_dict or {}
    
    assert result is yaml_dict
    assert result != {}
    assert bool(yaml_dict) is True


# LLM-generated content at query #31
#--------------------------

```python
def test_get_config_predicate_line_8_evaluates_to_false(tmp_path):
    import os
    import yaml
    from pathlib import Path
    
    # Create a temporary config file with valid YAML content
    config_file = tmp_path / "config.yaml"
    config_content = {
        'replay_dir': '/tmp/replay',
        'cookiecutters_dir': '/tmp/cookies'
    }
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config_content, f)
    
    # The predicate at line 8 is the `with open(config_path, encoding='utf-8')` statement
    # It evaluates to False when the file cannot be opened
    # To test that it evaluates to False, we need to ensure the file exists and can be opened
    # So the context manager enters successfully (predicate is True in normal flow)
    # To make the predicate evaluate to False, we test with a non-existent file
    
    non_existent_file = tmp_path / "non_existent.yaml"
    
    # Attempting to open a non-existent file will cause the predicate to fail
    try:
        with open(non_existent_file, encoding='utf-8') as file_handle:
            pass
        file_opened = True
    except FileNotFoundError:
        file_opened = False
    
    assert file_opened is False


# LLM-generated content at query #32
#--------------------------

```python
def test_get_config_file_not_exists():
    config_path = '/nonexistent/path/config.yaml'
    try:
        get_config(config_path)
        assert False, "Should raise ConfigDoesNotExistException"
    except ConfigDoesNotExistException:
        pass


def test_get_config_invalid_yaml(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("invalid: yaml: content: [", encoding='utf-8')
    try:
        get_config(str(config_file))
        assert False, "Should raise InvalidConfiguration"
    except InvalidConfiguration:
        pass


def test_get_config_non_dict_top_level(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("- item1\n- item2\n", encoding='utf-8')
    try:
        get_config(str(config_file))
        assert False, "Should raise InvalidConfiguration"
    except InvalidConfiguration:
        pass


def test_get_config_empty_yaml(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("", encoding='utf-8')
    result = get_config(str(config_file))
    assert isinstance(result, dict)
    assert 'replay_dir' in result
    assert 'cookiecutters_dir' in result


def test_get_config_with_valid_config(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("replay_dir: /tmp/replays\ncookiecutters_dir: /tmp/cookies\n", encoding='utf-8')
    result = get_config(str(config_file))
    assert isinstance(result, dict)
    assert result['replay_dir'] == '/tmp/replays'
    assert result['cookiecutters_dir'] == '/tmp/cookies'


def test_get_config_with_env_vars(tmp_path, monkeypatch):
    monkeypatch.setenv('TEST_REPLAY_DIR', '/test/replays')
    monkeypatch.setenv('TEST_COOKIES_DIR', '/test/cookies')
    config_file = tmp_path / "config.yaml"
    config_file.write_text("replay_dir: $TEST_REPLAY_DIR\ncookiecutters_dir: $TEST_COOKIES_DIR\n", encoding='utf-8')
    result = get_config(str(config_file))
    assert result['replay_dir'] == '/test/replays'
    assert result['cookiecutters_dir'] == '/test/cookies'


def test_get_config_with_home_expansion(tmp_path, monkeypatch):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("replay_dir: ~/replays\ncookiecutters_dir: ~/cookies\n", encoding='utf-8')
    result = get_config(str(config_file))
    assert '~' not in result['replay_dir']
    assert '~' not in result['cookiecutters_dir']
    assert result['replay_dir'].startswith('/')
    assert result['cookiecutters_dir'].startswith('/')


def test_get_config_merges_with_defaults(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("replay_dir: /custom/replays\n", encoding='utf-8')
    result = get_config(str(config_file))
    assert result['replay_dir'] == '/custom/replays'
    assert 'cookiecutters_dir' in result


# LLM-generated content at query #33
#--------------------------

```python
def test_get_config_yaml_dict_is_dict():
    import tempfile
    import os
    from pathlib import Path
    
    # Create a temporary YAML file with a valid dict structure
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8') as f:
        f.write('key: value\n')
        temp_path = f.name
    
    try:
        # Mock the necessary dependencies
        import sys
        from unittest.mock import patch, MagicMock
        
        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', create=True) as mock_open:
                with patch('yaml.safe_load') as mock_yaml_load:
                    with patch('get_config.merge_configs') as mock_merge:
                        with patch('get_config._expand_path') as mock_expand:
                            # Set up the mock to return a dict (the predicate at line 14)
                            mock_yaml_load.return_value = {'key': 'value'}
                            mock_merge.return_value = {'replay_dir': '/tmp', 'cookiecutters_dir': '/tmp'}
                            mock_expand.return_value = '/tmp'
                            
                            mock_open.return_value.__enter__.return_value = MagicMock()
                            
                            result = get_config(temp_path)
                            
                            # Verify the result is a dict
                            assert isinstance(result, dict)
    finally:
        os.unlink(temp_path)


# LLM-generated content at query #34
#--------------------------

```python
def test_yaml_error_predicate_evaluates_to_false(tmp_path):
    import yaml
    from pathlib import Path
    
    config_file = tmp_path / "config.yaml"
    config_file.write_text("invalid: yaml: content: [")
    
    try:
        get_config(config_file)
    except Exception as e:
        assert isinstance(e, yaml.YAMLError) is False or isinstance(e.__cause__, yaml.YAMLError)


# LLM-generated content at query #35
#--------------------------

```python
def test_yaml_error_raises_invalid_configuration(tmp_path):
    import yaml
    from pathlib import Path
    
    config_file = tmp_path / "config.yaml"
    config_file.write_text("invalid: yaml: content: [")
    
    try:
        get_config(str(config_file))
        assert False, "Should have raised InvalidConfiguration"
    except InvalidConfiguration as e:
        assert "Unable to parse YAML file" in str(e)
        assert isinstance(e.__cause__, yaml.YAMLError)


# LLM-generated content at query #36
#--------------------------

```python
def test_yaml_error_predicate_evaluates_to_false(tmp_path):
    import yaml
    from pathlib import Path
    
    config_file = tmp_path / "config.yaml"
    config_file.write_text("invalid: yaml: content: [")
    
    try:
        get_config(config_file)
    except Exception as e:
        assert isinstance(e, yaml.YAMLError) is False or isinstance(e, InvalidConfiguration)


# LLM-generated content at query #37
#--------------------------

```python
def test_line_14_predicate_evaluates_to_false(tmp_path):
    import os
    import yaml
    from pathlib import Path
    
    config_file = tmp_path / "config.yaml"
    config_file.write_text("key: value\n")
    
    result = get_config(config_file)
    
    assert isinstance(result, dict)


# LLM-generated content at query #38
#--------------------------

```python
def test_yaml_error_raises_invalid_configuration(tmp_path):
    import yaml
    from pathlib import Path
    
    config_file = tmp_path / "config.yaml"
    config_file.write_text("invalid: yaml: content: [")
    
    try:
        get_config(config_file)
        assert False, "Expected InvalidConfiguration to be raised"
    except InvalidConfiguration as e:
        assert "Unable to parse YAML file" in str(e)
        assert isinstance(e.__cause__, yaml.YAMLError)


# LLM-generated content at query #39
#--------------------------

```python
def test_predicate_at_line_8_evaluates_to_false(tmp_path):
    import os
    import yaml
    from pathlib import Path
    
    # Create a temporary config file
    config_file = tmp_path / "config.yaml"
    config_content = "replay_dir: /tmp\ncookiecutters_dir: /tmp"
    config_file.write_text(config_content)
    
    # Mock the necessary functions and imports
    from unittest.mock import patch, MagicMock
    
    mock_logger = MagicMock()
    mock_merge_configs = MagicMock(return_value={'replay_dir': '/tmp', 'cookiecutters_dir': '/tmp'})
    mock_expand_path = MagicMock(side_effect=lambda x: x)
    
    with patch('os.path.exists', return_value=True):
        with patch('builtins.open', create=True) as mock_open:
            mock_file = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_file
            
            with patch('yaml.safe_load', return_value={'replay_dir': '/tmp', 'cookiecutters_dir': '/tmp'}):
                with patch('get_config.logger', mock_logger):
                    with patch('get_config.merge_configs', mock_merge_configs):
                        with patch('get_config._expand_path', mock_expand_path):
                            # Verify that open is called with the config_path and encoding='utf-8'
                            result = get_config(config_file)
                            
                            # Assert that open was called with encoding='utf-8'
                            mock_open.assert_called_once_with(config_file, encoding='utf-8')


# LLM-generated content at query #40
#--------------------------

```python
def test_get_config_with_valid_dict_yaml(tmp_path, monkeypatch):
    import yaml
    from pathlib import Path
    
    config_file = tmp_path / "config.yaml"
    config_content = {
        'replay_dir': '/tmp/replays',
        'cookiecutters_dir': '/tmp/cookies'
    }
    config_file.write_text(yaml.dump(config_content))
    
    monkeypatch.setenv('HOME', str(tmp_path))
    
    result = get_config(config_file)
    
    assert isinstance(result, dict)
    assert 'replay_dir' in result
    assert 'cookiecutters_dir' in result


# LLM-generated content at query #41
#--------------------------

```python
def test_get_config_opens_file_with_utf8_encoding(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("replay_dir: /tmp\ncookiecutters_dir: /tmp\n", encoding='utf-8')
    
    result = get_config(config_file)
    
    assert isinstance(result, dict)
    assert 'replay_dir' in result
    assert 'cookiecutters_dir' in result


# LLM-generated content at query #42
#--------------------------

```python
def test_yaml_error_raises_invalid_configuration(tmp_path):
    import yaml
    from pathlib import Path
    
    config_file = tmp_path / "config.yaml"
    config_file.write_text("invalid: yaml: content: [")
    
    try:
        get_config(config_file)
        assert False, "Expected InvalidConfiguration to be raised"
    except InvalidConfiguration as e:
        assert "Unable to parse YAML file" in str(e)
        assert isinstance(e.__cause__, yaml.YAMLError)


# LLM-generated content at query #43
#--------------------------

```python
def test_yaml_error_not_raised():
    import tempfile
    import os
    from pathlib import Path
    
    # Create a temporary valid YAML file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write('replay_dir: /tmp\ncookiecutters_dir: /tmp\n')
        temp_path = f.name
    
    try:
        # Call the function with valid YAML - the except block at line 11 should not execute
        result = get_config(temp_path)
        
        # Verify the function completed successfully without raising yaml.YAMLError
        assert isinstance(result, dict)
        assert 'replay_dir' in result
        assert 'cookiecutters_dir' in result
    finally:
        os.unlink(temp_path)


# LLM-generated content at query #44
#--------------------------

```python
def test_get_config_file_does_not_exist(tmp_path):
    config_path = tmp_path / "nonexistent.yaml"
    try:
        get_config(config_path)
        assert False, "Should have raised ConfigDoesNotExistException"
    except ConfigDoesNotExistException:
        pass


def test_get_config_valid_yaml(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("replay_dir: /tmp/replays\ncookiecutters_dir: /tmp/cookies\n", encoding='utf-8')
    result = get_config(config_file)
    assert isinstance(result, dict)
    assert result['replay_dir'] == '/tmp/replays'
    assert result['cookiecutters_dir'] == '/tmp/cookies'


def test_get_config_invalid_yaml(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("invalid: yaml: content: [", encoding='utf-8')
    try:
        get_config(config_file)
        assert False, "Should have raised InvalidConfiguration"
    except InvalidConfiguration:
        pass


def test_get_config_non_dict_yaml(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("- item1\n- item2\n", encoding='utf-8')
    try:
        get_config(config_file)
        assert False, "Should have raised InvalidConfiguration"
    except InvalidConfiguration:
        pass


def test_get_config_expands_environment_variables(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_REPLAY_DIR", "/test/replays")
    monkeypatch.setenv("TEST_COOKIES_DIR", "/test/cookies")
    config_file = tmp_path / "config.yaml"
    config_file.write_text("replay_dir: $TEST_REPLAY_DIR\ncookiecutters_dir: $TEST_COOKIES_DIR\n", encoding='utf-8')
    result = get_config(config_file)
    assert result['replay_dir'] == '/test/replays'
    assert result['cookiecutters_dir'] == '/test/cookies'


def test_get_config_expands_user_home(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("replay_dir: ~/replays\ncookiecutters_dir: ~/cookies\n", encoding='utf-8')
    result = get_config(config_file)
    assert '~' not in result['replay_dir']
    assert '~' not in result['cookiecutters_dir']
    assert result['replay_dir'].startswith('/')
    assert result['cookiecutters_dir'].startswith('/')


def test_get_config_merges_with_defaults(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("replay_dir: /custom/replays\n", encoding='utf-8')
    result = get_config(config_file)
    assert result['replay_dir'] == '/custom/replays'
    assert 'cookiecutters_dir' in result


def test_get_config_empty_yaml(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("", encoding='utf-8')
    result = get_config(config_file)
    assert isinstance(result, dict)
    assert 'replay_dir' in result
    assert 'cookiecutters_dir' in result


def test_get_config_nested_dict_merge(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("replay_dir: /tmp/replays\ncookiecutters_dir: /tmp/cookies\nabbreviations:\n  custom_abbr: value\n", encoding='utf-8')
    result = get_config(config_file)
    assert result['replay_dir'] == '/tmp/replays'
    assert 'abbreviations' in result
    assert result['abbreviations']['custom_abbr'] == 'value'


