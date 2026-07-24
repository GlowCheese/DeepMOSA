####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_get_user_config_with_default_config():
    default_config = {'replay_dir': '/default/replay', 'cookiecutters_dir': '/default/cookiecutters'}
    result = get_user_config(default_config=default_config)
    assert result == default_config

def test_get_user_config_with_default_config_as_dict():
    custom_config = {'replay_dir': '/custom/replay', 'cookiecutters_dir': '/custom/cookiecutters'}
    result = get_user_config(default_config=custom_config)
    expected = {'replay_dir': '/custom/replay', 'cookiecutters_dir': '/custom/cookiecutters'}
    assert result == expected

def test_get_user_config_with_default_config_as_true():
    result = get_user_config(default_config=True)
    assert result == DEFAULT_CONFIG

def test_get_user_config_with_custom_config_file():
    custom_config_file = '/path/to/custom/config.yaml'
    result = get_user_config(config_file=custom_config_file)
    assert isinstance(result, dict)

def test_get_user_config_with_env_var():
    os.environ['COOKIECUTTER_CONFIG'] = '/path/to/env/config.yaml'
    result = get_user_config()
    assert isinstance(result, dict)

def test_get_user_config_with_default_user_config():
    result = get_user_config()
    assert isinstance(result, dict)


# LLM-generated content at query #2
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
    except ConfigDoesNotExistException as e:
        assert str(e) == f'Config file {config_path} does not exist.'

def test_get_config_with_invalid_yaml():
    config_path = "invalid_config.yaml"
    with open(config_path, "w") as f:
        f.write("invalid: yaml: file")
    try:
        get_config(config_path)
    except InvalidConfiguration as e:
        assert str(e) == f'Unable to parse YAML file {config_path}.'
    os.remove(config_path)

def test_get_config_with_non_dict_yaml():
    config_path = "non_dict_config.yaml"
    with open(config_path, "w") as f:
        f.write("- item1\n- item2")
    try:
        get_config(config_path)
    except InvalidConfiguration as e:
        assert str(e) == f'Top-level element of YAML file {config_path} should be an object.'
    os.remove(config_path)


# LLM-generated content at query #3
#--------------------------

```python
def test_predicate_at_line_40_evaluates_to_false():
    os.environ.pop('COOKIECUTTER_CONFIG', None)
    assert 'COOKIECUTTER_CONFIG' not in os.environ


# LLM-generated content at query #4
#--------------------------

```python
def test_get_user_config_default_config_true():
    config = get_user_config(default_config=True)
    assert config == DEFAULT_CONFIG

def test_get_user_config_default_config_dict():
    custom_config = {'replay_dir': '/custom/replay', 'cookiecutters_dir': '/custom/cookiecutters'}
    config = get_user_config(default_config=custom_config)
    expected_config = merge_configs(DEFAULT_CONFIG, custom_config)
    assert config == expected_config

def test_get_user_config_custom_config_file():
    custom_config_path = '/path/to/custom/config.yaml'
    config = get_user_config(config_file=custom_config_path)
    expected_config = get_config(custom_config_path)
    assert config == expected_config

def test_get_user_config_env_config_file(monkeypatch):
    env_config_path = '/path/to/env/config.yaml'
    monkeypatch.setenv('COOKIECUTTER_CONFIG', env_config_path)
    config = get_user_config()
    expected_config = get_config(env_config_path)
    assert config == expected_config

def test_get_user_config_default_user_config():
    config = get_user_config()
    if os.path.exists(USER_CONFIG_PATH):
        expected_config = get_config(USER_CONFIG_PATH)
    else:
        expected_config = copy.copy(DEFAULT_CONFIG)
    assert config == expected_config


# LLM-generated content at query #5
#--------------------------

```python
def test_predicate_at_line_40_evaluates_to_false():
    os.environ.pop('COOKIECUTTER_CONFIG', None)
    result = get_user_config()
    assert result == copy.copy(DEFAULT_CONFIG)


# LLM-generated content at query #6
#--------------------------

```python
def test_config_file_exists_returns_config_from_user_config_path():
    USER_CONFIG_PATH = "/path/to/user/config"
    os.path.exists = lambda path: path == USER_CONFIG_PATH
    result = get_user_config()
    assert result == get_config(USER_CONFIG_PATH)

def test_config_file_does_not_exist_returns_default_config():
    USER_CONFIG_PATH = "/path/to/user/config"
    os.path.exists = lambda path: False
    result = get_user_config()
    assert result == copy.copy(DEFAULT_CONFIG


# LLM-generated content at query #7
#--------------------------

```python
def test_predicate_at_line_43_evaluates_to_true_when_user_config_path_exists():
    import os
    import tempfile
    from unittest.mock import patch

    with tempfile.NamedTemporaryFile() as tmp:
        USER_CONFIG_PATH = tmp.name
        with patch('os.path.exists', return_value=True):
            assert os.path.exists(USER_CONFIG_PATH)


# LLM-generated content at query #8
#--------------------------

```
def test_predicate_at_line_40_evaluates_to_false():
    os.environ['COOKIECUTTER_CONFIG'] = 'some_path'
    try:
        get_user_config()
    except KeyError:
        assert False, "Predicate at line 40 should evaluate to False when COOKIECUTTER_CONFIG is set"
    finally:
        del os.environ['COOKIECUTTER_CONFIG']


# LLM-generated content at query #9
#--------------------------

```python
def test_get_user_config_default_config_true():
    result = get_user_config(default_config=True)
    expected = copy.copy(DEFAULT_CONFIG)
    assert result == expected


def test_get_user_config_default_config_dict():
    custom_config = {'replay_dir': '/custom/replay', 'cookiecutters_dir': '/custom/cookies'}
    result = get_user_config(default_config=custom_config)
    expected = merge_configs(DEFAULT_CONFIG, custom_config)
    assert result == expected


def test_get_user_config_custom_config_file():
    custom_config_path = '/path/to/custom/config.yaml'
    result = get_user_config(config_file=custom_config_path)
    expected = get_config(custom_config_path)
    assert result == expected


def test_get_user_config_env_config_file():
    env_config_path = '/path/to/env/config.yaml'
    os.environ['COOKIECUTTER_CONFIG'] = env_config_path
    result = get_user_config()
    expected = get_config(env_config_path)
    assert result == expected
    del os.environ['COOKIECUTTER_CONFIG']


def test_get_user_config_user_config_path():
    result = get_user_config()
    expected = get_config(USER_CONFIG_PATH) if os.path.exists(USER_CONFIG_PATH) else copy.copy(DEFAULT_CONFIG)
    assert result == expected


def test_get_user_config_user_config_not_found():
    result = get_user_config()
    expected = copy.copy(DEFAULT_CONFIG)
    assert result == expected


# LLM-generated content at query #10
#--------------------------

```python
def test_get_config_raises_exception_when_config_path_does_not_exist():
    config_path = "non_existent_path"
    try:
        get_config(config_path)
    except ConfigDoesNotExistException:
        pass
    else:
        assert False, "Expected ConfigDoesNotExistException to be raised"


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
    custom_config = {'replay_dir': '/custom/replay'}
    result = get_user_config(default_config=custom_config)
    expected = merge_configs(DEFAULT_CONFIG, custom_config)
    assert result == expected

def test_get_user_config_custom_config_file():
    custom_config_path = '/path/to/custom/config.yaml'
    result = get_user_config(config_file=custom_config_path)
    expected = get_config(custom_config_path)
    assert result == expected

def test_get_user_config_env_config_file(monkeypatch):
    env_config_path = '/path/to/env/config.yaml'
    monkeypatch.setenv('COOKIECUTTER_CONFIG', env_config_path)
    result = get_user_config()
    expected = get_config(env_config_path)
    assert result == expected

def test_get_user_config_user_config_path_exists(monkeypatch):
    monkeypatch.delenv('COOKIECUTTER_CONFIG', raising=False)
    monkeypatch.setattr(os.path, 'exists', lambda path: path == USER_CONFIG_PATH)
    result = get_user_config()
    expected = get_config(USER_CONFIG_PATH)
    assert result == expected

def test_get_user_config_default_config_fallback(monkeypatch):
    monkeypatch.delenv('COOKIECUTTER_CONFIG', raising=False)
    monkeypatch.setattr(os.path, 'exists', lambda path: False)
    result = get_user_config()
    assert result == DEFAULT_CONFIG


