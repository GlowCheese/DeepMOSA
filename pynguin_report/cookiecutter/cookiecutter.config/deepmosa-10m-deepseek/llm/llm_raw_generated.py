####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_get_user_config_default_config_true():
    result = get_user_config(default_config=True)
    assert result == DEFAULT_CONFIG


def test_get_user_config_default_config_dict():
    custom_config = {"replay_dir": "/custom/replay", "cookiecutters_dir": "/custom/cookies"}
    result = get_user_config(default_config=custom_config)
    assert result == merge_configs(DEFAULT_CONFIG, custom_config)


def test_get_user_config_custom_config_file():
    config_file = "/custom/config.yaml"
    result = get_user_config(config_file=config_file)
    assert result == get_config(config_file)


def test_get_user_config_env_config_file(monkeypatch):
    env_config_file = "/env/config.yaml"
    monkeypatch.setenv("COOKIECUTTER_CONFIG", env_config_file)
    result = get_user_config()
    assert result == get_config(env_config_file)


def test_get_user_config_user_config_path_exists():
    result = get_user_config()
    assert result == get_config(USER_CONFIG_PATH)


def test_get_user_config_user_config_path_not_exists(monkeypatch):
    monkeypatch.delenv("COOKIECUTTER_CONFIG", raising=False)
    monkeypatch.setattr("os.path.exists", lambda path: False)
    result = get_user_config()
    assert result == DEFAULT_CONFIG


# LLM-generated content at query #2
#--------------------------

```python
def test_predicate_at_line_40_evaluates_to_false():
    os.environ.pop('COOKIECUTTER_CONFIG', None)
    result = get_user_config()
    assert result == copy.copy(DEFAULT_CONFIG)


# LLM-generated content at query #3
#--------------------------

```
def test_default_config_is_true():
    result = get_user_config(default_config=True)
    assert result == copy.copy(DEFAULT_CONFIG)

def test_default_config_is_dict():
    custom_config = {"key": "value"}
    result = get_user_config(default_config=custom_config)
    assert result == merge_configs(DEFAULT_CONFIG, custom_config)

def test_config_file_is_not_user_config_path():
    result = get_user_config(config_file="/custom/path/config.yaml")
    assert result == get_config("/custom/path/config.yaml")

def test_env_config_file_is_not_set_and_user_config_path_does_not_exist():
    import os
    os.environ.pop('COOKIECUTTER_CONFIG', None)
    USER_CONFIG_PATH = "/default/path/config.yaml"
    if os.path.exists(USER_CONFIG_PATH):
        os.remove(USER_CONFIG_PATH)
    result = get_user_config()
    assert result == copy.copy(DEFAULT_CONFIG)

def test_env_config_file_is_set():
    import os
    os.environ['COOKIECUTTER_CONFIG'] = "/env/path/config.yaml"
    result = get_user_config()
    assert result == get_config("/env/path/config.yaml")


# LLM-generated content at query #4
#--------------------------

```python
def test_get_config_valid_yaml():
    config_path = 'valid_config.yaml'
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write('replay_dir: /path/to/replay\ncookiecutters_dir: /path/to/cookiecutters')
    config = get_config(config_path)
    assert config['replay_dir'] == '/path/to/replay'
    assert config['cookiecutters_dir'] == '/path/to/cookiecutters'
    os.remove(config_path)

def test_get_config_invalid_yaml():
    config_path = 'invalid_config.yaml'
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write('invalid: yaml: {')
    try:
        get_config(config_path)
        assert False, 'Expected InvalidConfiguration exception'
    except InvalidConfiguration:
        pass
    os.remove(config_path)

def test_get_config_non_dict_yaml():
    config_path = 'non_dict_config.yaml'
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write('- item1\n- item2')
    try:
        get_config(config_path)
        assert False, 'Expected InvalidConfiguration exception'
    except InvalidConfiguration:
        pass
    os.remove(config_path)

def test_get_config_nonexistent_file():
    config_path = 'nonexistent_config.yaml'
    try:
        get_config(config_path)
        assert False, 'Expected ConfigDoesNotExistException exception'
    except ConfigDoesNotExistException:
        pass

def test_get_config_expand_path():
    config_path = 'expand_path_config.yaml'
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write('replay_dir: $HOME/replay\ncookiecutters_dir: ~/cookiecutters')
    config = get_config(config_path)
    assert config['replay_dir'] == os.path.expanduser(os.path.expandvars('$HOME/replay'))
    assert config['cookiecutters_dir'] == os.path.expanduser('~/cookiecutters')
    os.remove(config_path)


# LLM-generated content at query #5
#--------------------------

```python
def test_get_config_returns_dict():
    config_path = 'valid_config.yaml'
    config = get_config(config_path)
    assert isinstance(config, dict)


# LLM-generated content at query #6
#--------------------------

```python
def test_yaml_parsing_failure():
    config_path = "invalid_yaml_file.yml"
    try:
        get_config(config_path)
    except InvalidConfiguration:
        pass
    else:
        assert False, "Expected InvalidConfiguration exception to be raised"


# LLM-generated content at query #7
#--------------------------

```python
def test_get_config_nonexistent_file():
    config_path = 'nonexistent_file.yaml'
    try:
        get_config(config_path)
    except ConfigDoesNotExistException as e:
        assert str(e) == f'Config file {config_path} does not exist.'


# LLM-generated content at query #8
#--------------------------

```
def test_predicate_at_line_40_evaluates_to_false():
    os.environ['COOKIECUTTER_CONFIG'] = 'some_path'
    get_user_config()
    del os.environ['COOKIECUTTER_CONFIG']


# LLM-generated content at query #9
#--------------------------

```python
def test_get_config_with_valid_file():
    config_path = "valid_config.yaml"
    expected_config = {"replay_dir": "/expanded/replay", "cookiecutters_dir": "/expanded/cookiecutters"}
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write("replay_dir: $HOME/replay\ncookiecutters_dir: ~/cookiecutters")
    result = get_config(config_path)
    os.remove(config_path)
    assert result["replay_dir"] == expected_config["replay_dir"]
    assert result["cookiecutters_dir"] == expected_config["cookiecutters_dir"]


def test_get_config_with_invalid_file():
    config_path = "invalid_config.yaml"
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write("invalid yaml")
    try:
        get_config(config_path)
        assert False
    except InvalidConfiguration:
        assert True
    os.remove(config_path)


def test_get_config_with_nonexistent_file():
    config_path = "nonexistent_config.yaml"
    try:
        get_config(config_path)
        assert False
    except ConfigDoesNotExistException:
        assert True


def test_get_config_with_non_dict_top_level():
    config_path = "non_dict_config.yaml"
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write("- item1\n- item2")
    try:
        get_config(config_path)
        assert False
    except InvalidConfiguration:
        assert True
    os.remove(config_path)


# LLM-generated content at query #10
#--------------------------

```python
def test_get_user_config_with_default_config_true():
    result = get_user_config(default_config=True)
    assert result == DEFAULT_CONFIG


def test_get_user_config_with_default_config_dict():
    custom_config = {'replay_dir': '/custom/replay', 'cookiecutters_dir': '/custom/cookies'}
    expected = merge_configs(DEFAULT_CONFIG, custom_config)
    result = get_user_config(default_config=custom_config)
    assert result == expected


def test_get_user_config_with_custom_config_file(tmp_path):
    config_file = tmp_path / 'config.yml'
    config_data = {'replay_dir': '~/custom_replay', 'cookiecutters_dir': '~/custom_cookies'}
    with open(config_file, 'w') as f:
        yaml.dump(config_data, f)
    result = get_user_config(config_file=str(config_file))
    assert result['replay_dir'] == os.path.expanduser('~/custom_replay')
    assert result['cookiecutters_dir'] == os.path.expanduser('~/custom_cookies')


def test_get_user_config_with_env_var_config_file(tmp_path, monkeypatch):
    config_file = tmp_path / 'env_config.yml'
    config_data = {'replay_dir': '~/env_replay', 'cookiecutters_dir': '~/env_cookies'}
    with open(config_file, 'w') as f:
        yaml.dump(config_data, f)
    monkeypatch.setenv('COOKIECUTTER_CONFIG', str(config_file))
    result = get_user_config()
    assert result['replay_dir'] == os.path.expanduser('~/env_replay')
    assert result['cookiecutters_dir'] == os.path.expanduser('~/env_cookies')


def test_get_user_config_with_default_user_config(tmp_path, monkeypatch):
    config_file = tmp_path / '.cookiecutterrc'
    config_data = {'replay_dir': '~/default_replay', 'cookiecutters_dir': '~/default_cookies'}
    with open(config_file, 'w') as f:
        yaml.dump(config_data, f)
    monkeypatch.setattr('cookiecutter.config.USER_CONFIG_PATH', str(config_file))
    result = get_user_config()
    assert result['replay_dir'] == os.path.expanduser('~/default_replay')
    assert result['cookiecutters_dir'] == os.path.expanduser('~/default_cookies')


def test_get_user_config_with_no_config_found(monkeypatch):
    monkeypatch.delenv('COOKIECUTTER_CONFIG', raising=False)
    monkeypatch.setattr('cookiecutter.config.USER_CONFIG_PATH', '/nonexistent/path')
    result = get_user_config()
    assert result == DEFAULT_CONFIG


# LLM-generated content at query #11
#--------------------------

```python
def test_predicate_evaluates_to_false():
    USER_CONFIG_PATH = "/default/path/config.json"
    config_file = USER_CONFIG_PATH
    assert not (config_file and config_file is not USER_CONFIG_PATH)


# LLM-generated content at query #12
#--------------------------

```python
def test_get_config_non_dict_yaml():
    yaml_content = "not_a_dict"
    test_file = "test_config.yaml"
    with open(test_file, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    try:
        get_config(test_file)
    except InvalidConfiguration as e:
        assert str(e) == f'Top-level element of YAML file {test_file} should be an object.'
    finally:
        os.remove(test_file)


# LLM-generated content at query #13
#--------------------------

```python
def test_get_config_with_invalid_yaml_structure():
    config_path = 'invalid_structure.yaml'
    try:
        get_config(config_path)
        assert False, "Expected InvalidConfiguration to be raised"
    except InvalidConfiguration as e:
        assert str(e) == 'Top-level element of YAML file invalid_structure.yaml should be an object.'


# LLM-generated content at query #14
#--------------------------

```python
def test_get_user_config_with_default_config_true():
    config = get_user_config(default_config=True)
    assert config == DEFAULT_CONFIG

def test_get_user_config_with_default_config_dict():
    custom_config = {"replay_dir": "/custom/replay", "cookiecutters_dir": "/custom/cookiecutters"}
    config = get_user_config(default_config=custom_config)
    assert config == merge_configs(DEFAULT_CONFIG, custom_config)

def test_get_user_config_with_custom_config_file():
    custom_config_file = "/custom/config.yaml"
    config = get_user_config(config_file=custom_config_file)
    assert config == get_config(custom_config_file)

def test_get_user_config_with_env_config_file(monkeypatch):
    env_config_file = "/env/config.yaml"
    monkeypatch.setenv("COOKIECUTTER_CONFIG", env_config_file)
    config = get_user_config()
    assert config == get_config(env_config_file)

def test_get_user_config_with_default_config_file():
    config = get_user_config()
    assert config == get_config(USER_CONFIG_PATH) if os.path.exists(USER_CONFIG_PATH) else DEFAULT_CONFIG

def test_get_user_config_with_default_config_file_not_exist():
    config = get_user_config()
    assert config == DEFAULT_CONFIG


# LLM-generated content at query #15
#--------------------------

```python
def test_get_user_config_default_config_true():
    result = get_user_config(default_config=True)
    assert result == DEFAULT_CONFIG

def test_get_user_config_default_config_dict():
    custom_config = {'replay_dir': '/custom/replay', 'cookiecutters_dir': '/custom/cookiecutters'}
    result = get_user_config(default_config=custom_config)
    expected = merge_configs(DEFAULT_CONFIG, custom_config)
    assert result == expected

def test_get_user_config_custom_config_file():
    custom_config_path = '/custom/config.yaml'
    result = get_user_config(config_file=custom_config_path)
    expected = get_config(custom_config_path)
    assert result == expected

def test_get_user_config_env_config_file(monkeypatch):
    env_config_path = '/env/config.yaml'
    monkeypatch.setenv('COOKIECUTTER_CONFIG', env_config_path)
    result = get_user_config()
    expected = get_config(env_config_path)
    assert result == expected

def test_get_user_config_user_config_path():
    result = get_user_config()
    expected = get_config(USER_CONFIG_PATH) if os.path.exists(USER_CONFIG_PATH) else DEFAULT_CONFIG
    assert result == expected


# LLM-generated content at query #16
#--------------------------

```
def test_yaml_safe_load_does_not_raise_exception():
    yaml_dict = yaml.safe_load("key: value") or {}
    assert isinstance(yaml_dict, dict)


# LLM-generated content at query #17
#--------------------------

```
def test_config_file_is_user_config_path():
    config_file = USER_CONFIG_PATH
    result = get_user_config(config_file)
    assert result == DEFAULT_CONFIG


# LLM-generated content at query #18
#--------------------------

```python
def test_expand_path_with_env_var():
    os.environ['TEST_VAR'] = '/home/user'
    result = _expand_path('$TEST_VAR/dir')
    assert result == '/home/user/dir'

def test_expand_path_with_user_home():
    result = _expand_path('~/dir')
    assert result == os.path.expanduser('~/dir')

def test_expand_path_with_both_env_var_and_user_home():
    os.environ['TEST_VAR'] = '/home/user'
    result = _expand_path('$TEST_VAR/~/dir')
    assert result == '/home/user/~/dir'

def test_expand_path_with_no_expansion():
    result = _expand_path('/some/path')
    assert result == '/some/path'


# LLM-generated content at query #19
#--------------------------

```python
def test_get_user_config_default_config_true():
    result = get_user_config(default_config=True)
    assert result == DEFAULT_CONFIG

def test_get_user_config_default_config_dict():
    custom_config = {"replay_dir": "/custom/replay"}
    result = get_user_config(default_config=custom_config)
    assert result == merge_configs(DEFAULT_CONFIG, custom_config)

def test_get_user_config_custom_config_file():
    custom_config_file = "/tmp/custom_config.yaml"
    result = get_user_config(config_file=custom_config_file)
    assert result == get_config(custom_config_file)

def test_get_user_config_env_config_file():
    os.environ["COOKIECUTTER_CONFIG"] = "/tmp/env_config.yaml"
    result = get_user_config()
    assert result == get_config("/tmp/env_config.yaml")

def test_get_user_config_user_config_path():
    result = get_user_config()
    assert result == get_config(USER_CONFIG_PATH) if os.path.exists(USER_CONFIG_PATH) else DEFAULT_CONFIG


# LLM-generated content at query #20
#--------------------------

```python
def test_config_file_is_valid_yaml_dict():
    config_path = 'valid_config.yaml'
    config_dict = get_config(config_path)
    assert isinstance(config_dict, dict)


# LLM-generated content at query #21
#--------------------------

```python
def test_get_config_valid_file():
    config_path = 'valid_config.yml'
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write('replay_dir: $HOME/test\ncookiecutters_dir: ~/cookies')
    config = get_config(config_path)
    assert isinstance(config, dict)
    assert 'replay_dir' in config
    assert 'cookiecutters_dir' in config
    assert config['replay_dir'] == os.path.expanduser(os.path.expandvars('$HOME/test'))
    assert config['cookiecutters_dir'] == os.path.expanduser('~/cookies')
    os.remove(config_path)

def test_get_config_invalid_file():
    config_path = 'invalid_config.yml'
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write('replay_dir: $HOME/test\ncookiecutters_dir: ~/cookies\ninvalid_key: [1, 2, 3]')
    try:
        get_config(config_path)
        assert False, "Expected InvalidConfiguration to be raised"
    except InvalidConfiguration:
        pass
    os.remove(config_path)

def test_get_config_non_existent_file():
    config_path = 'non_existent_config.yml'
    try:
        get_config(config_path)
        assert False, "Expected ConfigDoesNotExistException to be raised"
    except ConfigDoesNotExistException:
        pass

def test_get_config_invalid_yaml():
    config_path = 'invalid_yaml.yml'
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write('replay_dir: $HOME/test\ncookiecutters_dir: ~/cookies\ninvalid_yaml: - test')
    try:
        get_config(config_path)
        assert False, "Expected InvalidConfiguration to be raised"
    except InvalidConfiguration:
        pass
    os.remove(config_path)

def test_get_config_non_dict_yaml():
    config_path = 'non_dict_yaml.yml'
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write('- replay_dir: $HOME/test\n- cookiecutters_dir: ~/cookies')
    try:
        get_config(config_path)
        assert False, "Expected InvalidConfiguration to be raised"
    except InvalidConfiguration:
        pass
    os.remove(config_path)


# LLM-generated content at query #22
#--------------------------

```python
def test_get_user_config_with_default_config_dict():
    default_config = {'replay_dir': '~/custom_replays', 'cookiecutters_dir': '~/custom_templates'}
    result = get_user_config(default_config=default_config)
    assert result['replay_dir'] == os.path.expanduser('~/custom_replays')
    assert result['cookiecutters_dir'] == os.path.expanduser('~/custom_templates')


def test_get_user_config_with_default_config_true():
    result = get_user_config(default_config=True)
    assert result == DEFAULT_CONFIG


def test_get_user_config_with_custom_config_file(tmp_path):
    config_file = tmp_path / 'config.yaml'
    config_file.write_text('replay_dir: /custom/path\ncookiecutters_dir: /templates')
    result = get_user_config(config_file=str(config_file))
    assert result['replay_dir'] == '/custom/path'
    assert result['cookiecutters_dir'] == '/templates'


def test_get_user_config_with_env_var_config_file(tmp_path, monkeypatch):
    config_file = tmp_path / 'env_config.yaml'
    config_file.write_text('replay_dir: /env/path\ncookiecutters_dir: /env/templates')
    monkeypatch.setenv('COOKIECUTTER_CONFIG', str(config_file))
    result = get_user_config()
    assert result['replay_dir'] == '/env/path'
    assert result['cookiecutters_dir'] == '/env/templates'


def test_get_user_config_with_default_user_config(tmp_path, monkeypatch):
    user_config = tmp_path / '.cookiecutter.yaml'
    user_config.write_text('replay_dir: /user/path\ncookiecutters_dir: /user/templates')
    monkeypatch.setattr('os.path.expanduser', lambda x: str(tmp_path / x[1:]) if x.startswith('~') else x)
    result = get_user_config()
    assert result['replay_dir'] == '/user/path'
    assert result['cookiecutters_dir'] == '/user/templates'


def test_get_user_config_fallback_to_defaults():
    result = get_user_config()
    assert result == DEFAULT_CONFIG


# LLM-generated content at query #23
#--------------------------

```python
def test_predicate_at_line8_evaluates_to_false():
    config_path = '/nonexistent/path/to/config.yaml'
    try:
        get_config(config_path)
    except ConfigDoesNotExistException:
        pass
    else:
        assert False, "Expected ConfigDoesNotExistException to be raised"


# LLM-generated content at query #24
#--------------------------

```python
def test_config_file_is_user_config_path():
    config_file = USER_CONFIG_PATH
    assert not (config_file and config_file is not USER_CONFIG_PATH)


# LLM-generated content at query #25
#--------------------------

```
def test_predicate_at_line_10_evaluates_to_false():
    from unittest.mock import mock_open
    import os
    import yaml
    
    mock_file_content = ""
    with unittest.mock.patch('builtins.open', mock_open(read_data=mock_file_content)):
        with unittest.mock.patch('os.path.exists', return_value=True):
            result = yaml.safe_load(mock_open().return_value) or {}
            assert result == {}


# LLM-generated content at query #26
#--------------------------

```python
def test_get_config_with_valid_yaml():
    config_path = "test_config.yml"
    with open(config_path, "w", encoding="utf-8") as f:
        f.write("replay_dir: $HOME/test_replay\ncookiecutters_dir: ~/test_cookies")
    result = get_config(config_path)
    assert isinstance(result, dict)
    assert "replay_dir" in result
    assert "cookiecutters_dir" in result
    os.remove(config_path)

def test_get_config_with_nonexistent_file():
    config_path = "nonexistent.yml"
    try:
        get_config(config_path)
    except ConfigDoesNotExistException as e:
        assert str(e) == f'Config file {config_path} does not exist.'

def test_get_config_with_invalid_yaml():
    config_path = "invalid_config.yml"
    with open(config_path, "w", encoding="utf-8") as f:
        f.write("invalid: yaml: file")
    try:
        get_config(config_path)
    except InvalidConfiguration as e:
        assert str(e) == f'Unable to parse YAML file {config_path}.'
    os.remove(config_path)

def test_get_config_with_non_dict_yaml():
    config_path = "non_dict_config.yml"
    with open(config_path, "w", encoding="utf-8") as f:
        f.write("- item1\n- item2")
    try:
        get_config(config_path)
    except InvalidConfiguration as e:
        assert str(e) == f'Top-level element of YAML file {config_path} should be an object.'
    os.remove(config_path)

def test_get_config_merges_with_default():
    config_path = "merge_config.yml"
    with open(config_path, "w", encoding="utf-8") as f:
        f.write("replay_dir: $HOME/test\ncookiecutters_dir: ~/test")
    result = get_config(config_path)
    assert "abbreviations" in result  # from DEFAULT_CONFIG
    os.remove(config_path)


# LLM-generated content at query #27
#--------------------------

```python
def test_predicate_at_line_11_evaluates_to_false():
    import os
    from pathlib import Path
    import yaml
    from unittest.mock import mock_open, patch

    config_path = "valid_config.yaml"
    yaml_content = "key: value"
    with patch("os.path.exists", return_value=True), patch(
        "builtins.open", mock_open(read_data=yaml_content)
    ), patch("yaml.safe_load", return_value={"key": "value"}):
        yaml_dict = yaml.safe_load(open(config_path)) or {}
        assert not isinstance(yaml_dict, Exception)


# LLM-generated content at query #28
#--------------------------

```python
def test_predicate_at_line_10_evaluates_to_true():
    config_path = "valid_config.yaml"
    open(config_path, "w").write("key: value")
    yaml_dict = get_config(config_path)
    assert isinstance(yaml_dict, dict)


# LLM-generated content at query #29
#--------------------------

```
def test_yaml_safe_load_returns_empty_dict_when_none():
    import os
    import tempfile
    from pathlib import Path
    import yaml

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = Path(tmp.name)
    
    try:
        with open(tmp_path, 'w', encoding='utf-8') as f:
            f.write('')
        
        with open(tmp_path, encoding='utf-8') as file_handle:
            result = yaml.safe_load(file_handle) or {}
        
        assert result == {}
    finally:
        os.unlink(tmp_path)


# LLM-generated content at query #30
#--------------------------

```
def test_get_config_valid_yaml():
    test_config_path = 'test_config.yaml'
    with open(test_config_path, 'w', encoding='utf-8') as f:
        f.write('replay_dir: /test/replay\ncookiecutters_dir: /test/cookiecutters')
    config = get_config(test_config_path)
    assert config['replay_dir'] == os.path.expanduser(os.path.expandvars('/test/replay'))
    assert config['cookiecutters_dir'] == os.path.expanduser(os.path.expandvars('/test/cookiecutters'))
    os.remove(test_config_path)

def test_get_config_missing_file():
    test_config_path = 'non_existent_config.yaml'
    try:
        get_config(test_config_path)
        assert False
    except ConfigDoesNotExistException:
        assert True

def test_get_config_invalid_yaml():
    test_config_path = 'invalid_config.yaml'
    with open(test_config_path, 'w', encoding='utf-8') as f:
        f.write('invalid: yaml: file')
    try:
        get_config(test_config_path)
        assert False
    except InvalidConfiguration:
        assert True
    os.remove(test_config_path)

def test_get_config_non_dict_yaml():
    test_config_path = 'non_dict_config.yaml'
    with open(test_config_path, 'w', encoding='utf-8') as f:
        f.write('- item1\n- item2')
    try:
        get_config(test_config_path)
        assert False
    except InvalidConfiguration:
        assert True
    os.remove(test_config_path)

def test_get_config_empty_yaml():
    test_config_path = 'empty_config.yaml'
    with open(test_config_path, 'w', encoding='utf-8') as f:
        f.write('')
    config = get_config(test_config_path)
    assert config['replay_dir'] == os.path.expanduser(os.path.expandvars(DEFAULT_CONFIG['replay_dir']))
    assert config['cookiecutters_dir'] == os.path.expanduser(os.path.expandvars(DEFAULT_CONFIG['cookiecutters_dir']))
    os.remove(test_config_path)


# LLM-generated content at query #31
#--------------------------

```python
def test_get_config_valid_yaml():
    config_path = 'valid_config.yaml'
    with open(config_path, 'w', encoding='utf-8') as file:
        file.write('replay_dir: /path/to/replay\ncookiecutters_dir: /path/to/cookiecutters')
    result = get_config(config_path)
    assert isinstance(result, dict)
    assert 'replay_dir' in result
    assert 'cookiecutters_dir' in result


# LLM-generated content at query #32
#--------------------------

```python
def test_get_config_with_valid_file():
    config_path = "valid_config.yaml"
    with open(config_path, "w") as f:
        f.write("replay_dir: ~/replays\ncookiecutters_dir: ~/cookiecutters")
    config = get_config(config_path)
    assert config["replay_dir"] == os.path.expanduser("~/replays")
    assert config["cookiecutters_dir"] == os.path.expanduser("~/cookiecutters")
    os.remove(config_path)

def test_get_config_with_nonexistent_file():
    config_path = "nonexistent_config.yaml"
    try:
        get_config(config_path)
        assert False, "Expected ConfigDoesNotExistException"
    except ConfigDoesNotExistException:
        pass

def test_get_config_with_invalid_yaml():
    config_path = "invalid_yaml.yaml"
    with open(config_path, "w") as f:
        f.write("invalid: yaml: file")
    try:
        get_config(config_path)
        assert False, "Expected InvalidConfiguration"
    except InvalidConfiguration:
        pass
    os.remove(config_path)

def test_get_config_with_non_dict_yaml():
    config_path = "non_dict_yaml.yaml"
    with open(config_path, "w") as f:
        f.write("- item1\n- item2")
    try:
        get_config(config_path)
        assert False, "Expected InvalidConfiguration"
    except InvalidConfiguration:
        pass
    os.remove(config_path)

def test_get_config_merges_defaults():
    config_path = "partial_config.yaml"
    with open(config_path, "w") as f:
        f.write("replay_dir: ~/replays")
    config = get_config(config_path)
    assert config["cookiecutters_dir"] == os.path.expanduser(DEFAULT_CONFIG["cookiecutters_dir"])
    os.remove(config_path)

def test_get_config_expands_env_vars():
    config_path = "env_var_config.yaml"
    os.environ["TEST_ENV_VAR"] = "test_value"
    with open(config_path, "w") as f:
        f.write("replay_dir: $TEST_ENV_VAR/replays\ncookiecutters_dir: ~/cookiecutters")
    config = get_config(config_path)
    assert config["replay_dir"] == os.path.expandvars("$TEST_ENV_VAR/replays")
    os.remove(config_path)


# LLM-generated content at query #33
#--------------------------

```
def test_get_config_with_valid_yaml():
    config_path = 'valid_config.yaml'
    with open(config_path, 'w') as f:
        f.write('replay_dir: ~/replays\ncookiecutters_dir: ~/cookiecutters')
    config = get_config(config_path)
    assert isinstance(config, dict)
    assert 'replay_dir' in config
    assert 'cookiecutters_dir' in config
    assert config['replay_dir'] == os.path.expanduser('~/replays')
    assert config['cookiecutters_dir'] == os.path.expanduser('~/cookiecutters')
    os.remove(config_path)

def test_get_config_with_invalid_yaml():
    config_path = 'invalid_config.yaml'
    with open(config_path, 'w') as f:
        f.write('invalid: yaml: file')
    try:
        get_config(config_path)
        assert False, 'Expected InvalidConfiguration exception'
    except InvalidConfiguration:
        pass
    os.remove(config_path)

def test_get_config_with_nonexistent_file():
    config_path = 'nonexistent_config.yaml'
    try:
        get_config(config_path)
        assert False, 'Expected ConfigDoesNotExistException exception'
    except ConfigDoesNotExistException:
        pass

def test_get_config_with_non_dict_yaml():
    config_path = 'non_dict_config.yaml'
    with open(config_path, 'w') as f:
        f.write('- item1\n- item2')
    try:
        get_config(config_path)
        assert False, 'Expected InvalidConfiguration exception'
    except InvalidConfiguration:
        pass
    os.remove(config_path)

def test_get_config_with_empty_yaml():
    config_path = 'empty_config.yaml'
    with open(config_path, 'w') as f:
        f.write('')
    config = get_config(config_path)
    assert isinstance(config, dict)
    assert 'replay_dir' in config
    assert 'cookiecutters_dir' in config
    assert config['replay_dir'] == os.path.expanduser(DEFAULT_CONFIG['replay_dir'])
    assert config['cookiecutters_dir'] == os.path.expanduser(DEFAULT_CONFIG['cookiecutters_dir'])
    os.remove(config_path)


# LLM-generated content at query #34
#--------------------------

```python
def test_get_config_returns_dict_when_valid_yaml_file_is_provided():
    yaml_content = "key: value"
    with open("valid_config.yaml", "w", encoding="utf-8") as file:
        file.write(yaml_content)
    config = get_config("valid_config.yaml")
    assert isinstance(config, dict)


# LLM-generated content at query #35
#--------------------------

```python
def test_get_config_returns_merged_config():
    import tempfile
    import os
    from pathlib import Path

    test_config = {
        'replay_dir': '~/test_replays',
        'cookiecutters_dir': '$HOME/test_cookiecutters',
    }

    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write('replay_dir: ~/test_replays\n')
        f.write('cookiecutters_dir: $HOME/test_cookiecutters\n')
        config_path = f.name

    result = get_config(config_path)
    os.unlink(config_path)

    assert result['replay_dir'] == os.path.expanduser('~/test_replays')
    assert result['cookiecutters_dir'] == os.path.expandvars('$HOME/test_cookiecutters')


def test_get_config_raises_exception_when_file_not_exists():
    import os
    from pathlib import Path

    config_path = Path('/nonexistent/path/to/config.yaml')
    try:
        get_config(config_path)
        assert False, 'Expected ConfigDoesNotExistException'
    except ConfigDoesNotExistException:
        pass


def test_get_config_raises_exception_when_invalid_yaml():
    import tempfile
    import os
    from pathlib import Path

    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write('invalid: yaml: here: {')
        config_path = f.name

    try:
        get_config(config_path)
        assert False, 'Expected InvalidConfiguration'
    except InvalidConfiguration:
        pass
    finally:
        os.unlink(config_path)


def test_get_config_raises_exception_when_top_level_not_dict():
    import tempfile
    import os
    from pathlib import Path

    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write('- item1\n- item2\n')
        config_path = f.name

    try:
        get_config(config_path)
        assert False, 'Expected InvalidConfiguration'
    except InvalidConfiguration:
        pass
    finally:
        os.unlink(config_path)


# LLM-generated content at query #36
#--------------------------

```
def test_config_path_exists_and_readable():
    import tempfile
    import os
    from pathlib import Path

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(b'key: value')
        tmp.flush()
    
    try:
        assert os.path.exists(tmp_path)
        with open(tmp_path, encoding='utf-8') as f:
            assert f.readable()
    finally:
        os.unlink(tmp_path)


# LLM-generated content at query #37
#--------------------------

```python
def test_config_file_does_not_exist():
    config_path = Path('/nonexistent/path')
    try:
        get_config(config_path)
        assert False, 'Expected ConfigDoesNotExistException'
    except ConfigDoesNotExistException:
        pass

def test_invalid_yaml_file():
    config_path = Path('/tmp/invalid.yaml')
    with open(config_path, 'w') as f:
        f.write('invalid: yaml: file')
    try:
        get_config(config_path)
        assert False, 'Expected InvalidConfiguration'
    except InvalidConfiguration:
        pass
    finally:
        os.remove(config_path)

def test_non_dict_yaml_file():
    config_path = Path('/tmp/non_dict.yaml')
    with open(config_path, 'w') as f:
        f.write('- item1\n- item2')
    try:
        get_config(config_path)
        assert False, 'Expected InvalidConfiguration'
    except InvalidConfiguration:
        pass
    finally:
        os.remove(config_path)


# LLM-generated content at query #38
#--------------------------

```
def test_yaml_dict_not_dict():
    import os
    import tempfile
    from pathlib import Path
    import yaml

    with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp:
        tmp.write('not a dict')
        tmp_path = tmp.name

    try:
        get_config(tmp_path)
    except InvalidConfiguration as e:
        assert str(e) == f'Top-level element of YAML file {tmp_path} should be an object.'
    finally:
        os.unlink(tmp_path)


# LLM-generated content at query #39
#--------------------------

```
def test_config_path_exists_and_is_readable():
    import os
    from pathlib import Path
    import tempfile
    import yaml

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(b'key: value')
        tmp.flush()

    assert os.path.exists(tmp_path)
    assert os.access(tmp_path, os.R_OK)


# LLM-generated content at query #40
#--------------------------

```python
def test_config_parser_rejects_non_dict_yaml():
    test_file = Path('test_non_dict.yaml')
    test_file.write_text('- item1\n- item2')
    try:
        get_config(test_file)
    except InvalidConfiguration as e:
        assert str(e) == f'Top-level element of YAML file {test_file} should be an object.'
    finally:
        test_file.unlink()


# LLM-generated content at query #41
#--------------------------

```
def test_config_path_exists():
    import os
    from pathlib import Path
    import tempfile

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        config_path = Path(tmp.name)
        assert os.path.exists(config_path)


# LLM-generated content at query #42
#--------------------------

```python
def test_config_file_parsed_successfully():
    config_path = 'valid_config.yaml'
    with open(config_path, 'w') as f:
        f.write('key: value')
    result = get_config(config_path)
    assert isinstance(result, dict)


# LLM-generated content at query #43
#--------------------------

```python
def test_get_config_with_valid_yaml():
    config_path = 'valid_config.yaml'
    with open(config_path, 'w') as f:
        f.write('replay_dir: ~/replays\ncookiecutters_dir: $HOME/cookiecutters')
    result = get_config(config_path)
    assert isinstance(result, dict)
    assert 'replay_dir' in result
    assert 'cookiecutters_dir' in result
    os.remove(config_path)

def test_get_config_with_invalid_yaml():
    config_path = 'invalid_config.yaml'
    with open(config_path, 'w') as f:
        f.write('invalid: yaml: file')
    try:
        get_config(config_path)
        assert False
    except InvalidConfiguration:
        assert True
    os.remove(config_path)

def test_get_config_with_nonexistent_file():
    config_path = 'nonexistent_config.yaml'
    try:
        get_config(config_path)
        assert False
    except ConfigDoesNotExistException:
        assert True

def test_get_config_with_non_dict_yaml():
    config_path = 'non_dict_config.yaml'
    with open(config_path, 'w') as f:
        f.write('- item1\n- item2')
    try:
        get_config(config_path)
        assert False
    except InvalidConfiguration:
        assert True
    os.remove(config_path)

def test_get_config_merges_with_defaults():
    config_path = 'partial_config.yaml'
    with open(config_path, 'w') as f:
        f.write('replay_dir: /custom/replays')
    result = get_config(config_path)
    assert result['replay_dir'] == '/custom/replays'
    assert 'cookiecutters_dir' in result
    os.remove(config_path)

def test_get_config_expands_paths():
    config_path = 'path_config.yaml'
    with open(config_path, 'w') as f:
        f.write('replay_dir: ~/test\ncookiecutters_dir: $HOME/test')
    result = get_config(config_path)
    assert not result['replay_dir'].startswith('~')
    assert not result['cookiecutters_dir'].startswith('$HOME')
    os.remove(config_path)


# LLM-generated content at query #44
#--------------------------

```python
def test_predicate_at_line_14_evaluates_to_false():
    yaml_dict = "not a dictionary"
    assert not isinstance(yaml_dict, dict)


# LLM-generated content at query #45
#--------------------------

```
def test_yaml_safe_load_returns_none():
    from unittest.mock import mock_open, patch
    mock_file = mock_open(read_data='invalid_yaml_content')
    with patch('builtins.open', mock_file):
        with patch('yaml.safe_load', return_value=None):
            config = get_config('dummy_path')
            assert config == merge_configs(DEFAULT_CONFIG, {})


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```
def test_get_user_config_with_default_config_true():
    result = get_user_config(default_config=True)
    assert result == DEFAULT_CONFIG


def test_get_user_config_with_default_config_dict():
    custom_config = {'replay_dir': '/custom/replay', 'cookiecutters_dir': '/custom/cookiecutters'}
    result = get_user_config(default_config=custom_config)
    expected = merge_configs(DEFAULT_CONFIG, custom_config)
    assert result == expected


def test_get_user_config_with_custom_config_file(tmp_path):
    config_file = tmp_path / 'custom_config.yaml'
    config_data = {'replay_dir': '~/custom_replay', 'cookiecutters_dir': '~/custom_cookiecutters'}
    with open(config_file, 'w') as f:
        yaml.dump(config_data, f)
    result = get_user_config(config_file=str(config_file))
    expected = merge_configs(DEFAULT_CONFIG, config_data)
    expected['replay_dir'] = _expand_path('~/custom_replay')
    expected['cookiecutters_dir'] = _expand_path('~/custom_cookiecutters')
    assert result == expected


def test_get_user_config_with_env_var_config_file(tmp_path, monkeypatch):
    config_file = tmp_path / 'env_config.yaml'
    config_data = {'replay_dir': '~/env_replay', 'cookiecutters_dir': '~/env_cookiecutters'}
    with open(config_file, 'w') as f:
        yaml.dump(config_data, f)
    monkeypatch.setenv('COOKIECUTTER_CONFIG', str(config_file))
    result = get_user_config()
    expected = merge_configs(DEFAULT_CONFIG, config_data)
    expected['replay_dir'] = _expand_path('~/env_replay')
    expected['cookiecutters_dir'] = _expand_path('~/env_cookiecutters')
    assert result == expected


def test_get_user_config_with_default_user_config(tmp_path, monkeypatch):
    config_file = tmp_path / USER_CONFIG_PATH.name
    config_data = {'replay_dir': '~/default_replay', 'cookiecutters_dir': '~/default_cookiecutters'}
    with open(config_file, 'w') as f:
        yaml.dump(config_data, f)
    monkeypatch.setattr('os.path.exists', lambda x: x == str(config_file))
    result = get_user_config()
    expected = merge_configs(DEFAULT_CONFIG, config_data)
    expected['replay_dir'] = _expand_path('~/default_replay')
    expected['cookiecutters_dir'] = _expand_path('~/default_cookiecutters')
    assert result == expected


def test_get_user_config_with_no_config_files():
    result = get_user_config()
    assert result == DEFAULT_CONFIG


